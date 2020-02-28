# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from torch.nn import functional as F

from meshrcnn.structures.mask import batch_crop_masks_within_box


def mask_rcnn_loss(pred_mask_logits, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
        and groundtruth masks for visualization
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = batch_crop_masks_within_box(
            instances_per_image.gt_masks, instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, gt_masks

    gt_masks = cat(gt_masks, dim=0)
    assert gt_masks.numel() > 0, gt_masks.shape

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    # Note that here we allow gt_masks to be float as well
    # (depend on the implementation of rasterize())
    mask_accurate = (pred_mask_logits > 0.5) == (gt_masks > 0.5)
    mask_accuracy = mask_accurate.nonzero().size(0) / mask_accurate.numel()
    get_event_storage().put_scalar("mask_rcnn/accuracy", mask_accuracy)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )
    return mask_loss, gt_masks
