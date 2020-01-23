# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import meshrcnn.utils.shape as shape_utils


def batch_crop_voxels_within_box(voxels, boxes, Ks, voxel_side_len):
    """
    Batched version of :func:`crop_voxel_within_box`.

    Args:
        voxels (VoxelInstances): store N voxels for an image
        boxes (Tensor): store N boxes corresponding to the masks.
        Ks (Tensor): store N camera matrices
        voxel_side_len (int): the size of the voxel.

    Returns:
        Tensor: A byte tensor of shape (N, voxel_side_len, voxel_side_len, voxel_side_len),
        where N is the number of predicted boxes for this image.
    """
    device = boxes.device
    im_sizes = Ks[:, 1:] * 2.0
    voxels_tensor = torch.stack(voxels.data, 0)
    zranges = torch.stack(
        [voxels_tensor[:, :, 2].min(1)[0], voxels_tensor[:, :, 2].max(1)[0]], dim=1
    )
    cub3D = shape_utils.box2D_to_cuboid3D(zranges, Ks, boxes.clone(), im_sizes)
    txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)
    x, y, z = voxels_tensor.split(1, dim=2)
    xz = torch.cat([x, z], dim=2)
    yz = torch.cat([y, z], dim=2)
    pxz = txz(xz)
    pyz = tyz(yz)
    cropped_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)
    results = [
        verts2voxel(cropped_vert, [voxel_side_len] * 3).permute(2, 0, 1)
        for cropped_vert in cropped_verts
    ]

    if len(results) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(results, dim=0).to(device=device)


class VoxelInstances:
    """
    Class to hold voxels of varying dimensions to interface with Instances
    """

    def __init__(self, voxels):
        assert isinstance(voxels, list)
        assert torch.is_tensor(voxels[0])
        self.data = voxels

    def to(self, device):
        to_voxels = [voxel.to(device) for voxel in self]
        return VoxelInstances(to_voxels)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_data = [self.data[item]]
        else:
            # advanced indexing on a single dimension
            selected_data = []
            if isinstance(item, torch.Tensor) and (
                item.dtype == torch.uint8 or item.dtype == torch.bool
            ):
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_data.append(self.data[i])
        return VoxelInstances(selected_data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}) ".format(len(self))
        return s


def downsample(vox_in, n, use_max=True):
    """
    Downsample a 3-d tensor n times
    Inputs:
      - vox_in (Tensor): HxWxD tensor
      - n (int): number of times to downsample each dimension
      - use_max (bool): use maximum value when downsampling. If set to False
                        the mean value is used.
    Output:
      - vox_out (Tensor): (H/n)x(W/n)x(D/n) tensor
    """
    dimy = vox_in.size(0) // n
    dimx = vox_in.size(1) // n
    dimz = vox_in.size(2) // n
    vox_out = torch.zeros((dimy, dimx, dimz))
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                subx = x * n
                suby = y * n
                subz = z * n
                subvox = vox_in[suby : suby + n, subx : subx + n, subz : subz + n]
                if use_max:
                    vox_out[y, x, z] = torch.max(subvox)
                else:
                    vox_out[y, x, z] = torch.mean(subvox)
    return vox_out


def verts2voxel(verts, voxel_size):
    def valid_coords(x, y, z, vx_size):
        Hv, Wv, Zv = vx_size
        indx = (x >= 0) * (x < Wv)
        indy = (y >= 0) * (y < Hv)
        indz = (z >= 0) * (z < Zv)
        return indx * indy * indz

    Hv, Wv, Zv = voxel_size
    # create original voxel of size VxVxV
    orig_voxel = torch.zeros((Hv, Wv, Zv), dtype=torch.float32)

    x = (verts[:, 0] + 1) * (Wv - 1) / 2
    x = x.long()
    y = (verts[:, 1] + 1) * (Hv - 1) / 2
    y = y.long()
    z = (verts[:, 2] + 1) * (Zv - 1) / 2
    z = z.long()

    keep = valid_coords(x, y, z, voxel_size)
    x = x[keep]
    y = y[keep]
    z = z[keep]

    orig_voxel[y, x, z] = 1.0

    # align with image coordinate system
    flip_idx = torch.tensor(list(range(Hv)[::-1]))
    orig_voxel = orig_voxel.index_select(0, flip_idx)
    flip_idx = torch.tensor(list(range(Wv)[::-1]))
    orig_voxel = orig_voxel.index_select(1, flip_idx)
    return orig_voxel


def normalize_verts(verts):
    # centering and normalization
    min, _ = torch.min(verts, 0)
    min_x, min_y, min_z = min
    max, _ = torch.max(verts, 0)
    max_x, max_y, max_z = max
    x_ctr = (min_x + max_x) / 2.0
    y_ctr = (min_y + max_y) / 2.0
    z_ctr = (min_z + max_z) / 2.0
    x_scale = 2.0 / (max_x - min_x)
    y_scale = 2.0 / (max_y - min_y)
    z_scale = 2.0 / (max_z - min_z)
    verts[:, 0] = (verts[:, 0] - x_ctr) * x_scale
    verts[:, 1] = (verts[:, 1] - y_ctr) * y_scale
    verts[:, 2] = (verts[:, 2] - z_ctr) * z_scale
    return verts
