# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

ROI_Z_HEAD_REGISTRY = Registry("ROI_Z_HEAD")


@ROI_Z_HEAD_REGISTRY.register()
class FastRCNNFCHead(nn.Module):
    """
    A head with several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_fc: the number of fc layers
            fc_dim: the dimension of the fc layers
        """
        super().__init__()

        # fmt: off
        num_fc          = cfg.MODEL.ROI_Z_HEAD.NUM_FC
        fc_dim          = cfg.MODEL.ROI_Z_HEAD.FC_DIM
        cls_agnostic    = cfg.MODEL.ROI_Z_HEAD.CLS_AGNOSTIC_Z_REG
        num_classes     = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # fmt: on

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("z_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        num_z_reg_classes = 1 if cls_agnostic else num_classes
        self.z_pred = nn.Linear(fc_dim, num_z_reg_classes)

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        nn.init.normal_(self.z_pred.weight, std=0.001)
        nn.init.constant_(self.z_pred.bias, 0)

    def forward(self, x):
        x = x.view(x.shape[0], np.prod(x.shape[1:]))
        for layer in self.fcs:
            x = F.relu(layer(x))
        x = self.z_pred(x)
        return x

    @property
    def output_size(self):
        return self._output_size


def z_rcnn_loss(z_pred, instances, src_boxes, loss_weight=1.0, smooth_l1_beta=0.0):
    """
    Compute the z_pred loss.

    Args:
        z_pred (Tensor): A tensor of shape (B, C) or (B, 1) for class-specific or class-agnostic,
            where B is the total number of foreground regions in all images, C is the number of foreground classes,
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_z = z_pred.size(1) == 1
    total_num = z_pred.size(0)

    gt_classes = []
    gt_dz = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_z:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_dz.append(instances_per_image.gt_dz)

    if len(gt_dz) == 0:
        return z_pred.sum() * 0

    gt_dz = cat(gt_dz, dim=0)
    assert gt_dz.numel() > 0
    src_heights = src_boxes[:, 3] - src_boxes[:, 1]
    dz = torch.log(gt_dz / src_heights)

    if cls_agnostic_z:
        z_pred = z_pred[:, 0]
    else:
        indices = torch.arange(total_num)
        gt_classes = cat(gt_classes, dim=0)
        z_pred = z_pred[indices, gt_classes]

    loss_z_reg = smooth_l1_loss(z_pred, dz, smooth_l1_beta, reduction="sum")
    loss_z_reg = loss_weight * loss_z_reg / gt_classes.numel()
    return loss_z_reg


def z_rcnn_inference(z_pred, pred_instances):
    cls_agnostic = z_pred.size(1) == 1

    if not cls_agnostic:
        num_z = z_pred.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_z, device=class_pred.device)
        z_pred = z_pred[indices, class_pred][:, None]

    z_pred = torch.clamp(z_pred, max=math.log(1000.0 / 16))
    z_pred = torch.exp(z_pred)

    # The multiplication with the heights of the boxes will happen at eval time
    # See appendix for more.

    num_boxes_per_image = [len(i) for i in pred_instances]
    z_pred = z_pred.split(num_boxes_per_image, dim=0)

    for z_reg, instances in zip(z_pred, pred_instances):
        instances.pred_dz = z_reg


def build_z_head(cfg, input_shape):
    name = cfg.MODEL.ROI_Z_HEAD.NAME
    return ROI_Z_HEAD_REGISTRY.get(name)(cfg, input_shape)
