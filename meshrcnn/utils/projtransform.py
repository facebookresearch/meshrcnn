# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


class ProjectiveTransform(object):
    """
    Projective Transformation in PyTorch:
    Follows a similar design to skimage.ProjectiveTransform
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py#L494
    The implementation assumes batched representations,
    so every tensor is assumed to be of shape batch x dim1 x dim2 x etc.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            # default to an identity transform
            matrix = torch.eye(3).view(1, 3, 3)
        if matrix.ndim != 3 and matrix.shape[-1] != 3 and matrix.shape[-2] != 3:
            raise ValueError("Shape of transformation matrix should be Bx3x3")
        self.params = matrix

    @property
    def _inv_matrix(self):
        return torch.inverse(self.params)

    def _apply_mat(self, coords, matrix):
        """
        Applies matrix transformation
        Input:
            coords: FloatTensor of shape BxNx2
            matrix: FloatTensor of shape Bx3x3
        Returns:
            new_coords: FloatTensor of shape BxNx2
        """
        if coords.shape[0] != matrix.shape[0]:
            raise ValueError("Mismatch in the batch dimension")
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError("Input tensors should be of shape BxNx2")

        # append 1s, shape: BxNx2 -> BxNx3
        src = torch.cat(
            [
                coords,
                torch.ones(
                    (coords.shape[0], coords.shape[1], 1), device=coords.device, dtype=torch.float32
                ),
            ],
            dim=2,
        )
        dst = torch.bmm(matrix, src.transpose(1, 2)).transpose(1, 2)
        # rescale to homogeneous coordinates
        dst[:, :, 0] /= dst[:, :, 2]
        dst[:, :, 1] /= dst[:, :, 2]

        return dst[:, :, :2]

    def __call__(self, coords):
        """Apply forward transformation.
        Input:
            coords: FloatTensor of shape BxNx2
        Output:
            coords: FloateTensor of shape BxNx2
        """
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        """Apply inverse transformation.
        Input:
            coords: FloatTensor of shape BxNx2
        Output:
            coords: FloatTensor of shape BxNx2
        """
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst, method="svd"):
        """
        Estimates the matrix to transform src to dst.
        Input:
            src: FloatTensor of shape BxNx2
            dst: FloatTensor of shape BxNx2
            method: Specifies the method to solve the linear system
        """
        if src.shape != dst.shape:
            raise ValueError("src and dst tensors but be of same shape")
        if src.ndim != 3 or src.shape[-1] != 2:
            raise ValueError("Input should be of shape BxNx2")
        device = src.device
        batch = src.shape[0]

        # Center and normalize image points for better numerical stability.
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = torch.zeros((batch, 3, 3), device=device)
            return False

        xs = src[:, :, 0]
        ys = src[:, :, 1]
        xd = dst[:, :, 0]
        yd = dst[:, :, 1]
        rows = src.shape[1]

        # params: a0, a1, a2, b0, b1, b2, c0, c1, (c3=1)
        A = torch.zeros((batch, rows * 2, 9), device=device, dtype=torch.float32)
        A[:, :rows, 0] = xs
        A[:, :rows, 1] = ys
        A[:, :rows, 2] = 1
        A[:, :rows, 6] = -xd * xs
        A[:, :rows, 7] = -xd * ys
        A[:, rows:, 3] = xs
        A[:, rows:, 4] = ys
        A[:, rows:, 5] = 1
        A[:, rows:, 6] = -yd * xs
        A[:, rows:, 7] = -yd * ys
        A[:, :rows, 8] = xd
        A[:, rows:, 8] = yd

        if method == "svd":
            A = A.cpu()  # faster computation in cpu
            # Solve for the nullspace of the constraint matrix.
            _, _, V = torch.svd(A, some=False)
            V = V.transpose(1, 2)

            H = torch.ones((batch, 9), device=device, dtype=torch.float32)
            H[:, :-1] = -V[:, -1, :-1] / V[:, -1, -1].view(-1, 1)
            H = H.reshape(batch, 3, 3)
            # H[:, 2, 2] = 1.0
        elif method == "least_sqr":
            A = A.cpu()  # faster computation in cpu
            # Least square solution
            x, _ = torch.solve(-A[:, :, -1].view(-1, 1), A[:, :, :-1])
            H = torch.cat([-x, torch.ones((1, 1), dtype=x.dtype, device=device)])
            H = H.reshape(3, 3)
        elif method == "inv":
            # x = inv(A'A)*A'*b
            invAtA = torch.inverse(torch.mm(A[:, :-1].t(), A[:, :-1]))
            Atb = torch.mm(A[:, :-1].t(), -A[:, -1].view(-1, 1))
            x = torch.mm(invAtA, Atb)
            H = torch.cat([-x, torch.ones((1, 1), dtype=x.dtype, device=device)])
            H = H.reshape(3, 3)
        else:
            raise ValueError("method {} undefined".format(method))

        # De-center and de-normalize
        self.params = torch.bmm(torch.bmm(torch.inverse(dst_matrix), H), src_matrix)
        return True


def _center_and_normalize_points(points):
    """Center and normalize points.
    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).
    Inputs:
        points: FloatTensor of shape BxNx2 of the coordinates of the image points.
    Outputs:
        matrix: FloatTensor of shape Bx3x3 of the transformation matrix to obtain
                the new points.
        new_points: FloatTensor of shape BxNx2 of the transformed image points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.
    """
    device = points.device
    centroid = torch.mean(points, 1, keepdim=True)

    rms = torch.sqrt(torch.sum((points - centroid) ** 2.0, dim=(1, 2)) / points.shape[1])

    norm_factor = torch.sqrt(torch.tensor([2.0], device=device)) / rms

    matrix = torch.zeros((points.shape[0], 3, 3), dtype=torch.float32, device=device)
    matrix[:, 0, 0] = norm_factor
    matrix[:, 0, 2] = -norm_factor * centroid[:, 0, 0]
    matrix[:, 1, 1] = norm_factor
    matrix[:, 1, 2] = -norm_factor * centroid[:, 0, 1]
    matrix[:, 2, 2] = 1.0

    # matrix = torch.tensor(
    #     [
    #        [norm_factor, 0.0, -norm_factor * centroid[0]],
    #        [0.0, norm_factor, -norm_factor * centroid[1]],
    #        [0.0, 0.0, 1.0],
    #    ], device=device, dtype=torch.float32)

    pointsh = torch.cat(
        [
            points,
            torch.ones((points.shape[0], points.shape[1], 1), device=device, dtype=torch.float32),
        ],
        dim=2,
    )

    new_pointsh = torch.bmm(matrix, pointsh.transpose(1, 2)).transpose(1, 2)

    new_points = new_pointsh[:, :, :2]
    new_points[:, :, 0] /= new_pointsh[:, :, 2]
    new_points[:, :, 1] /= new_pointsh[:, :, 2]

    return matrix, new_points
