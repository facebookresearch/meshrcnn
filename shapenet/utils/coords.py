# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
""" Utilities for working with different 3D coordinate systems """

import copy
import math
import torch
from pytorch3d.structures import Meshes

SHAPENET_MIN_ZMIN = 0.67
SHAPENET_MAX_ZMAX = 0.92

SHAPENET_AVG_ZMIN = 0.77
SHAPENET_AVG_ZMAX = 0.90


def get_blender_intrinsic_matrix(N=None):
    """
    This is the (default) matrix that blender uses to map from camera coordinates
    to normalized device coordinates. We can extract it from Blender like this:

    import bpy
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    K = camera.calc_matrix_camera(
         render.resolution_x,
         render.resolution_y,
         render.pixel_aspect_x,
         render.pixel_aspect_y)
    """
    K = [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
    K = torch.tensor(K)
    if N is not None:
        K = K.view(1, 4, 4).expand(N, 4, 4)
    return K


def blender_ndc_to_world(verts):
    """
    Inverse operation to projecting by the Blender intrinsic operation above.
    In other words, the following should hold:

    K = get_blender_intrinsic_matrix()
    verts == blender_ndc_to_world(project_verts(verts, K))
    """
    xx, yy, zz = verts.unbind(dim=1)
    a1, a2, a3 = 2.1875, 2.1875, -1.002002
    b1, b2 = -0.2002002, -1.0
    z = b1 / (b2 * zz - a3)
    y = (b2 / a2) * (z * yy)
    x = (b2 / a1) * (z * xx)
    out = torch.stack([x, y, z], dim=1)
    return out


def voxel_to_world(meshes):
    """
    When predicting voxels, we operate in a [-1, 1]^3 coordinate space where the
    intrinsic matrix has already been applied, the y-axis has been flipped to
    to align with the image plane, and the z-axis has been rescaled so the min/max
    z values in the dataset correspond to -1 / 1. This function undoes these
    transformations, and projects a Meshes from voxel-space into world space.

    TODO: This projection logic is tightly coupled to the MeshVox Dataset;
    they should maybe both be refactored?

    Input:
    - meshes: Meshes in voxel coordinate system

    Output:
    - meshes: Meshes in world coordinate system
    """
    verts = meshes.verts_packed()
    x, y, z = verts.unbind(dim=1)

    zmin, zmax = SHAPENET_MIN_ZMIN, SHAPENET_MAX_ZMAX
    m = 2.0 / (zmax - zmin)
    b = -2.0 * zmin / (zmax - zmin) - 1

    y = -y
    z = (z - b) / m
    verts = torch.stack([x, y, z], dim=1)
    verts = blender_ndc_to_world(verts)

    verts_list = list(verts.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    faces_list = copy.deepcopy(meshes.faces_list())
    meshes_world = Meshes(verts=verts_list, faces=faces_list)

    return meshes_world


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.

    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin

    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT


def rotate_verts(RT, verts):
    """
    Inputs:
    - RT: (N, 4, 4) array of extrinsic matrices
    - verts: (N, V, 3) array of vertex positions
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    R = RT[:, :3, :3]
    verts_rot = verts.bmm(R.transpose(1, 2))
    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot


def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix

    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices

    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj
