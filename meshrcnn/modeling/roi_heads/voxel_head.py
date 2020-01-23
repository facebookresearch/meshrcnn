# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

from meshrcnn.structures.voxel import batch_crop_voxels_within_box

ROI_VOXEL_HEAD_REGISTRY = Registry("ROI_VOXEL_HEAD")


def voxel_rcnn_loss(pred_voxel_logits, instances, loss_weight=1.0):
    """
    Compute the voxel prediction loss defined in the Mesh R-CNN paper.

    Args:
        pred_voxel_logits (Tensor): A tensor of shape (B, C, D, H, W) or (B, 1, D, H, W)
            for class-specific or class-agnostic, where B is the total number of predicted voxels
            in all images, C is the number of foreground classes, and D, H, W are the depth,
            height and width of the voxel predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_voxel_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        loss_weight (float): A float to multiply the loss with.

    Returns:
        voxel_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_voxel = pred_voxel_logits.size(1) == 1
    total_num_voxels = pred_voxel_logits.size(0)
    voxel_side_len = pred_voxel_logits.size(2)
    assert pred_voxel_logits.size(2) == pred_voxel_logits.size(
        3
    ), "Voxel prediction must be square!"
    assert pred_voxel_logits.size(2) == pred_voxel_logits.size(
        4
    ), "Voxel prediction must be square!"

    gt_classes = []
    gt_voxel_logits = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_voxel:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_voxels = instances_per_image.gt_voxels
        gt_K = instances_per_image.gt_K
        gt_voxel_logits_per_image = batch_crop_voxels_within_box(
            gt_voxels, instances_per_image.proposal_boxes.tensor, gt_K, voxel_side_len
        ).to(device=pred_voxel_logits.device)
        gt_voxel_logits.append(gt_voxel_logits_per_image)

    if len(gt_voxel_logits) == 0:
        return pred_voxel_logits.sum() * 0, gt_voxel_logits

    gt_voxel_logits = cat(gt_voxel_logits, dim=0)
    assert gt_voxel_logits.numel() > 0, gt_voxel_logits.shape

    if cls_agnostic_voxel:
        pred_voxel_logits = pred_voxel_logits[:, 0]
    else:
        indices = torch.arange(total_num_voxels)
        gt_classes = cat(gt_classes, dim=0)
        pred_voxel_logits = pred_voxel_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    # Note that here we allow gt_voxel_logits to be float as well
    # (depend on the implementation of rasterize())
    voxel_accurate = (pred_voxel_logits > 0.5) == (gt_voxel_logits > 0.5)
    voxel_accuracy = voxel_accurate.nonzero().size(0) / voxel_accurate.numel()
    get_event_storage().put_scalar("voxel_rcnn/accuracy", voxel_accuracy)

    voxel_loss = F.binary_cross_entropy_with_logits(
        pred_voxel_logits, gt_voxel_logits, reduction="mean"
    )
    voxel_loss = voxel_loss * loss_weight
    return voxel_loss, gt_voxel_logits


def voxel_rcnn_inference(pred_voxel_logits, pred_instances):
    """
    Convert pred_voxel_logits to estimated foreground probability voxels while also
    extracting only the voxels for the predicted classes in pred_instances. For each
    predicted box, the voxel of the same class is attached to the instance by adding a
    new "pred_voxels" field to pred_instances.

    Args:
        pred_voxel_logits (Tensor): A tensor of shape (B, C, D, H, W) or (B, 1, D, H, W)
            for class-specific or class-agnostic, where B is the total number of predicted voxels
            in all images, C is the number of foreground classes, and D, H, W are the depth, height
            and width of the voxel predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_voxels" field storing a voxel of size (D, H,
            W) for predicted class. Note that the voxels are returned as a soft (non-quantized)
            voxels the resolution predicted by the network; post-processing steps are left
            to the caller.
    """
    cls_agnostic_voxel = pred_voxel_logits.size(1) == 1

    if cls_agnostic_voxel:
        voxel_probs_pred = pred_voxel_logits.sigmoid()
    else:
        # Select voxels corresponding to the predicted classes
        num_voxels = pred_voxel_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_voxels, device=class_pred.device)
        voxel_probs_pred = pred_voxel_logits[indices, class_pred][:, None].sigmoid()
    # voxel_probs_pred.shape: (B, 1, D, H, W)

    num_boxes_per_image = [len(i) for i in pred_instances]
    voxel_probs_pred = voxel_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(voxel_probs_pred, pred_instances):
        instances.pred_voxels = prob  # (1, D, H, W)


@ROI_VOXEL_HEAD_REGISTRY.register()
class VoxelRCNNConvUpsampleHead(nn.Module):
    """
    A voxel head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape):
        super(VoxelRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes        = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims          = cfg.MODEL.ROI_VOXEL_HEAD.CONV_DIM
        self.norm          = cfg.MODEL.ROI_VOXEL_HEAD.NORM
        num_conv           = cfg.MODEL.ROI_VOXEL_HEAD.NUM_CONV
        input_channels     = input_shape.channels
        cls_agnostic_voxel = cfg.MODEL.ROI_VOXEL_HEAD.CLS_AGNOSTIC_VOXEL
        # fmt: on

        self.conv_norm_relus = []
        self.num_depth = cfg.MODEL.ROI_VOXEL_HEAD.NUM_DEPTH
        self.num_classes = 1 if cls_agnostic_voxel else num_classes

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("voxel_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.predictor = Conv2d(
            conv_dims, self.num_classes * self.num_depth, kernel_size=1, stride=1, padding=0
        )

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for voxel prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        # reshape from (N, CD, H, W) to (N, C, D, H, W)
        x = x.reshape(x.size(0), self.num_classes, self.num_depth, x.size(2), x.size(3))
        return x


def build_voxel_head(cfg, input_shape):
    name = cfg.MODEL.ROI_VOXEL_HEAD.NAME
    return ROI_VOXEL_HEAD_REGISTRY.get(name)(cfg, input_shape)
