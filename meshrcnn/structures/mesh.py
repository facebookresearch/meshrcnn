# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytorch3d.structures import Meshes

import meshrcnn.utils.shape as shape_utils


def batch_crop_meshes_within_box(meshes, boxes, Ks):
    """
    Batched version of :func:`crop_mesh_within_box`.

    Args:
        mesh (MeshInstances): store N meshes for an image
        boxes (Tensor): store N boxes corresponding to the meshes.
        Ks (Tensor): store N camera matrices

    Returns:
        Meshes: A Meshes structure of N meshes where N is the number of
                predicted boxes for this image.
    """
    device = boxes.device
    im_sizes = Ks[:, 1:] * 2.0
    verts = torch.stack([mesh[0] for mesh in meshes], dim=0)
    zranges = torch.stack([verts[:, :, 2].min(1)[0], verts[:, :, 2].max(1)[0]], dim=1)
    cub3D = shape_utils.box2D_to_cuboid3D(zranges, Ks, boxes.clone(), im_sizes)
    txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)
    x, y, z = verts.split(1, dim=2)
    xz = torch.cat([x, z], dim=2)
    yz = torch.cat([y, z], dim=2)
    pxz = txz(xz)
    pyz = tyz(yz)
    new_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)

    # align to image
    new_verts[:, :, 0] = -new_verts[:, :, 0]
    new_verts[:, :, 1] = -new_verts[:, :, 1]

    verts_list = [new_verts[i] for i in range(boxes.shape[0])]
    faces_list = [mesh[1] for mesh in meshes]

    return Meshes(verts=verts_list, faces=faces_list).to(device=device)


class MeshInstances:
    """
    Class to hold meshes of varying topology to interface with Instances
    """

    def __init__(self, meshes):
        assert isinstance(meshes, list)
        assert torch.is_tensor(meshes[0][0])
        assert torch.is_tensor(meshes[0][1])
        self.data = meshes

    def to(self, device):
        to_meshes = [(mesh[0].to(device), mesh[1].to(device)) for mesh in self]
        return MeshInstances(to_meshes)

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
        return MeshInstances(selected_data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}) ".format(len(self))
        return s
