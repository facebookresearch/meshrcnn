# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from torch.nn import functional as F

from shapenet.utils.coords import project_verts


class MeshRefinementHead(nn.Module):
    def __init__(self, cfg):
        super(MeshRefinementHead, self).__init__()

        # fmt: off
        input_channels  = cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS
        self.num_stages = cfg.MODEL.MESH_HEAD.NUM_STAGES
        hidden_dim      = cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM
        stage_depth     = cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS
        graph_conv_init = cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else hidden_dim
            stage = MeshRefinementStage(
                input_channels, vert_feat_dim, hidden_dim, stage_depth, gconv_init=graph_conv_init
            )
            self.stages.append(stage)

    def forward(self, img_feats, meshes, P=None, subdivide=False):
        """
        Args:
          img_feats (tensor): Tensor of shape (N, C, H, W) giving image features,
                              or a list of such tensors.
          meshes (Meshes): Meshes class of N meshes
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
          subdivide (bool): Flag whether to subdivice the mesh after refinement

        Returns:
          output_meshes (list of Meshes): A list with S Meshes, where S is the
                                          number of refinement stages
        """
        output_meshes = []
        vert_feats = None
        for i, stage in enumerate(self.stages):
            meshes, vert_feats = stage(img_feats, meshes, vert_feats, P)
            output_meshes.append(meshes)
            if subdivide and i < self.num_stages - 1:
                subdivide = SubdivideMeshes()
                meshes, vert_feats = subdivide(meshes, feats=vert_feats)
        return output_meshes


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        """
        Args:
          img_feat_dim (int): Dimension of features we will get from vert_align
          vert_feat_dim (int): Dimension of vert_feats we will receive from the
                               previous stage; can be 0
          hidden_dim (int): Output dimension for graph-conv layers
          stage_depth (int): Number of graph-conv layers to use
          gconv_init (int): Specifies weight initialization for graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.vert_offset = nn.Linear(hidden_dim + 3, 3)

        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

    def forward(self, img_feats, meshes, vert_feats=None, P=None):
        """
        Args:
          img_feats (tensor): Features from the backbone
          meshes (Meshes): Initial meshes which will get refined
          vert_feats (tensor): Features from the previous refinement stage
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
        """
        # Project verts if we are making predictions in world space
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()

        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()

        # flip y coordinate
        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        factor = torch.tensor([1, -1, 1], device=device, dtype=dtype).view(1, 1, 3)
        vert_pos_padded = vert_pos_padded * factor
        # Get features from the image
        vert_align_feats = vert_align(img_feats, vert_pos_padded)
        vert_align_feats = _padded_to_packed(vert_align_feats, verts_padded_to_packed_idx)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        # Prepare features for first graph conv layer
        first_layer_feats = [vert_align_feats, vert_pos_packed]
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)
        vert_feats = torch.cat(first_layer_feats, dim=1)

        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, meshes.edges_packed()))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

        # Predict a new mesh by offsetting verts
        vert_offsets = torch.tanh(self.vert_offset(vert_feats))
        meshes_out = meshes.offset_verts(vert_offsets)

        return meshes_out, vert_feats_nopos


def _padded_to_packed(x, idx):
    """
    Convert features from padded to packed.

    Args:
      x: (N, V, D)
      idx: LongTensor of shape (VV,)

    Returns:
      feats_packed: (VV, D)
    """

    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
    x_packed = x.view(-1, D).gather(0, idx)
    return x_packed
