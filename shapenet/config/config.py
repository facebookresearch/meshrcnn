# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
def get_shapenet_cfg():

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = "resnet50"
    cfg.MODEL.VOXEL_ON = False
    cfg.MODEL.MESH_ON = False

    # ------------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CHECKPOINT = ""  # path to checkpoint

    # ------------------------------------------------------------------------ #
    # Voxel Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.VOXEL_HEAD = CN()
    # The number of convs in the voxel head and the number of channels
    cfg.MODEL.VOXEL_HEAD.NUM_CONV = 0
    cfg.MODEL.VOXEL_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers. Options: "" (no norm), "GN"
    cfg.MODEL.VOXEL_HEAD.NORM = ""
    # The number of depth channels for the predicted voxels
    cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE = 28
    cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.VOXEL_HEAD.CUBIFY_THRESH = 0.0
    # voxel only iterations
    cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS = 100

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.MESH_HEAD = CN()
    cfg.MODEL.MESH_HEAD.NAME = "VoxMeshHead"
    # Numer of stages
    cfg.MODEL.MESH_HEAD.NUM_STAGES = 1
    cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT = "normal"
    # Mesh sampling
    cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES = 5000
    cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES = 5000
    # loss weights
    cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT = 1.0
    # Init ico_sphere level (only for when voxel_on is false)
    cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL = -1

    # ------------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------------ #
    cfg.SOLVER = CN()
    cfg.SOLVER.LR_SCHEDULER_NAME = "constant"  # {'constant', 'cosine'}
    cfg.SOLVER.BATCH_SIZE = 32
    cfg.SOLVER.BATCH_SIZE_EVAL = 8
    cfg.SOLVER.NUM_EPOCHS = 25
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.OPTIMIZER = "adam"  # {'sgd', 'adam'}
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 24949  # in iters
    cfg.SOLVER.LOGGING_PERIOD = 50  # in iters
    # stable training
    cfg.SOLVER.SKIP_LOSS_THRESH = 50.0
    cfg.SOLVER.LOSS_SKIP_GAMMA = 0.9

    # ------------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------------ #
    cfg.DATASETS = CN()
    cfg.DATASETS.NAME = "shapenet"

    # ------------------------------------------------------------------------ #
    # Misc options
    # ------------------------------------------------------------------------ #
    # Directory where output files are written
    cfg.OUTPUT_DIR = "./output"

    return cfg
