# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def get_meshrcnn_cfg_defaults(cfg):
    """
    Customize the detectron2 cfg to include some new keys and default values
    for Mesh R-CNN
    """

    cfg.MODEL.VOXEL_ON = False
    cfg.MODEL.MESH_ON = False
    cfg.MODEL.ZPRED_ON = False
    cfg.MODEL.VIS_MINIBATCH = False  # visualize minibatches

    # aspect ratio grouping has no difference in performance
    # but might reduce memory by a little bit
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = False

    # ------------------------------------------------------------------------ #
    # Z Predict Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_Z_HEAD = CN()
    cfg.MODEL.ROI_Z_HEAD.NAME = "FastRCNNFCHead"
    cfg.MODEL.ROI_Z_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_Z_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_Z_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_Z_HEAD.POOLER_SAMPLING_RATIO = 2
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_Z_HEAD.POOLER_TYPE = "ROIAlign"
    # Whether to use class agnostic for z regression
    cfg.MODEL.ROI_Z_HEAD.CLS_AGNOSTIC_Z_REG = False
    # Default weight on (dz) for normalizing z regression targets
    cfg.MODEL.ROI_Z_HEAD.Z_REG_WEIGHT = 5.0
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    cfg.MODEL.ROI_Z_HEAD.SMOOTH_L1_BETA = 0.0

    # ------------------------------------------------------------------------ #
    # Voxel Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_VOXEL_HEAD = CN()
    cfg.MODEL.ROI_VOXEL_HEAD.NAME = "VoxelRCNNConvUpsampleHead"
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_VOXEL_HEAD.POOLER_TYPE = "ROIAlign"
    # Whether to use class agnostic for voxel prediction
    cfg.MODEL.ROI_VOXEL_HEAD.CLS_AGNOSTIC_VOXEL = False
    # The number of convs in the voxel head and the number of channels
    cfg.MODEL.ROI_VOXEL_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_VOXEL_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers. Options: "" (no norm), "GN"
    cfg.MODEL.ROI_VOXEL_HEAD.NORM = ""
    # The number of depth channels for the predicted voxels
    cfg.MODEL.ROI_VOXEL_HEAD.NUM_DEPTH = 28
    cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_VOXEL_HEAD.CUBIFY_THRESH = 0.0

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.ROI_MESH_HEAD = CN()
    cfg.MODEL.ROI_MESH_HEAD.NAME = "MeshRCNNGraphConvHead"
    cfg.MODEL.ROI_MESH_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MESH_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_MESH_HEAD.POOLER_TYPE = "ROIAlign"
    # Numer of stages
    cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES = 1
    cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT = "normal"
    # Mesh sampling
    cfg.MODEL.ROI_MESH_HEAD.GT_NUM_SAMPLES = 5000
    cfg.MODEL.ROI_MESH_HEAD.PRED_NUM_SAMPLES = 5000
    # loss weights
    cfg.MODEL.ROI_MESH_HEAD.CHAMFER_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MESH_HEAD.NORMALS_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_MESH_HEAD.EDGE_LOSS_WEIGHT = 1.0
    # coord thresh
    cfg.MODEL.ROI_MESH_HEAD.GT_COORD_THRESH = 0.0
    # Init ico_sphere level (only for when voxel_on is false)
    cfg.MODEL.ROI_MESH_HEAD.ICO_SPHERE_LEVEL = -1

    return cfg
