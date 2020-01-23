# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .lr_schedule import ConstantLR, WarmupCosineLR


def build_lr_scheduler(cfg, optimizer):
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "constant":
        return ConstantLR(optimizer)
    elif name == "cosine":
        return WarmupCosineLR(
            optimizer,
            total_iters=cfg.SOLVER.COMPUTED_MAX_ITERS,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmpup_factor=cfg.SOLVER.WARMUP_FACTOR,
        )


def build_optimizer(cfg, model):
    # TODO add weight decay?
    name = cfg.SOLVER.OPTIMIZER
    lr = cfg.SOLVER.BASE_LR
    momentum = cfg.SOLVER.MOMENTUM
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
