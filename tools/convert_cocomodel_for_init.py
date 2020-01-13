#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
Convert coco model for init. Remove class specific heads, optimizer and scheduler
so that this model can be used for pre-training
"""
import argparse

import torch


# TODO(gkioxari) delete this
def main_delete():
    import detectron2

    print("Loading model")
    dirpath = "/mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/gkioxari/20190821/test_train_f132759862/e2e_mesh_rcnn_R_50_FPN_1x.yaml.18_22_01.k8CXY0CX/output/inference/pix3d_clean_train/instances_predictions.pth"
    data = torch.load(dirpath)
    print("Writing detections")
    with open("/tmp/output/train_detections.txt", "w") as f:
        f.write("image_id x y w h\n")
        for i in range(len(data)):
            image_id = data[i]["image_id"]
            boxes = data[i]["instances"].pred_boxes.tensor
            for j in range(boxes.shape[0]):
                x = int(boxes[j, 0])
                y = int(boxes[j, 1])
                w = int(boxes[j, 2])
                h = int(boxes[j, 3])
                f.write("%d %d %d %d %d\n" % (image_id, x, y, w, h))


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
