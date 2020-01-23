#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
Convert coco model for init. Remove class specific heads, optimizer and scheduler
so that this model can be used for pre-training
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert models for init")
    parser.add_argument(
        "--model-file", default="", dest="modelfile", metavar="FILE", help="path to model", type=str
    )
    parser.add_argument(
        "--output-file",
        default="",
        dest="outputfile",
        metavar="FILE",
        help="path to model",
        type=str,
    )

    args = parser.parse_args()

    model = torch.load(args.modelfile)
    # pop the optimizer
    model.pop("optimizer")
    # pop the scheduler
    model.pop("scheduler")
    # pop the iteration
    model.pop("iteration")
    # pop the class specific weights from the coco pretrained model
    heads = [
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
        "roi_heads.mask_head.predictor.weight",
        "roi_heads.mask_head.predictor.bias",
    ]
    for head in heads:
        model["model"].pop(head)
    torch.save(model, args.outputfile)


if __name__ == "__main__":
    main()
