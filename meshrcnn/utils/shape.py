# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
from detectron2.utils.file_io import PathManager
from scipy import io as sio

from meshrcnn.utils.projtransform import ProjectiveTransform


def cuboid3D_to_unitbox3D(cub3D):
    device = cub3D.device
    dst = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], device=device, dtype=torch.float32
    )
    dst = dst.view(1, 4, 2).expand(cub3D.shape[0], -1, -1)
    # for (x,z) plane
    txz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack([cub3D[:, 0, 0], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 0, 1], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 2, 0], cub3D[:, 4, 1]], dim=1),
            torch.stack([cub3D[:, 2, 1], cub3D[:, 4, 1]], dim=1),
        ],
        dim=1,
    )
    if not txz.estimate(src, dst):
        raise ValueError("Estimate failed")
    # for (y,z) plane
    tyz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack([cub3D[:, 1, 0], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 1, 1], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 3, 0], cub3D[:, 4, 1]], dim=1),
            torch.stack([cub3D[:, 3, 1], cub3D[:, 4, 1]], dim=1),
        ],
        dim=1,
    )
    if not tyz.estimate(src, dst):
        raise ValueError("Estimate failed")
    return txz, tyz


def box2D_to_cuboid3D(zranges, Ks, boxes, im_sizes):
    device = boxes.device
    if boxes.shape[0] != Ks.shape[0] != zranges.shape[0]:
        raise ValueError("Ks, boxes and zranges must have the same batch dimension")
    if zranges.shape[1] != 2:
        raise ValueError("zrange must have two entries per example")
    w, h = im_sizes.t()
    sx, px, py = Ks.t()
    sy = sx
    x1, y1, x2, y2 = boxes.t()
    # transform 2d box from image coordinates to world coordinates
    x1 = w - 1 - x1 - px
    y1 = h - 1 - y1 - py
    x2 = w - 1 - x2 - px
    y2 = h - 1 - y2 - py

    cub3D = torch.zeros((boxes.shape[0], 5, 2), device=device, dtype=torch.float32)
    for i in range(2):
        z = zranges[:, i]
        x3D_min = x2 * z / sx
        x3D_max = x1 * z / sx
        y3D_min = y2 * z / sy
        y3D_max = y1 * z / sy
        cub3D[:, i * 2 + 0, 0] = x3D_min
        cub3D[:, i * 2 + 0, 1] = x3D_max
        cub3D[:, i * 2 + 1, 0] = y3D_min
        cub3D[:, i * 2 + 1, 1] = y3D_max
    cub3D[:, 4, 0] = zranges[:, 0]
    cub3D[:, 4, 1] = zranges[:, 1]
    return cub3D


def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = verts.clone().t()
    if R is not None:
        assert R.dim() == 2
        assert R.size(0) == 3 and R.size(1) == 3
        rot_verts = torch.mm(R, rot_verts)
    if t is not None:
        assert t.dim() == 1
        assert t.size(0) == 3
        rot_verts = rot_verts + t.unsqueeze(1)
    rot_verts = rot_verts.t()
    return rot_verts


def read_voxel(voxelfile):
    """
    Reads voxel and transforms it in the form of verts
    """
    with PathManager.open(voxelfile, "rb") as f:
        voxel = sio.loadmat(f)["voxel"]
    voxel = np.rot90(voxel, k=3, axes=(1, 2))
    verts = np.argwhere(voxel > 0).astype(np.float32, copy=False)

    # centering and normalization
    min_x = np.min(verts[:, 0])
    max_x = np.max(verts[:, 0])
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    min_z = np.min(verts[:, 2])
    max_z = np.max(verts[:, 2])
    verts[:, 0] = verts[:, 0] - (max_x + min_x) / 2
    verts[:, 1] = verts[:, 1] - (max_y + min_y) / 2
    verts[:, 2] = verts[:, 2] - (max_z + min_z) / 2
    scale = np.sqrt(np.max(np.sum(verts ** 2, axis=1))) * 2
    verts /= scale
    verts = torch.tensor(verts, dtype=torch.float32)

    return verts
