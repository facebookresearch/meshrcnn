# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F


def crop_mask_within_box(mask, box, mask_size):
    """
    Crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        mask (Tensor): A tensor mask image.
        box: 4 elements
        mask_size (int):

    Returns:
        Tensor: ByteTensor of shape (mask_size, mask_size)
    """
    # 1. Crop mask
    roi = box.clone().int()
    cropped_mask = mask[roi[1] : roi[3], roi[0] : roi[2]]

    # 2. Resize mask
    cropped_mask = cropped_mask.unsqueeze(0).unsqueeze(0)
    cropped_mask = F.interpolate(cropped_mask, size=(mask_size, mask_size), mode="bilinear")
    cropped_mask = cropped_mask.squeeze(0).squeeze(0)

    # 3. Binarize
    cropped_mask = (cropped_mask > 0).float()

    return cropped_mask


def batch_crop_masks_within_box(masks, boxes, mask_side_len):
    """
    Batched version of :func:`crop_mask_within_box`.

    Args:
        masks (Masks): store N masks for an image in 2D array format.
        boxes (Tensor): store N boxes corresponding to the masks.
        mask_side_len (int): the size of the mask.

    Returns:
        Tensor: A byte tensor of shape (N, mask_side_len, mask_side_len), where
            N is the number of predicted boxes for this image.
    """
    device = boxes.device
    # Put boxes on the CPU, as the representation for masks is not efficient
    # GPU-wise (possibly several small tensors for representing a single instance mask)
    boxes = boxes.to(torch.device("cpu"))
    masks = masks.to(torch.device("cpu"))

    results = [crop_mask_within_box(mask, box, mask_side_len) for mask, box in zip(masks, boxes)]

    if len(results) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(results, dim=0).to(device=device)
