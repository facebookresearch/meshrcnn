# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os

__all__ = ["default_argument_parser"]


def default_argument_parser():
    """
    Create a parser.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="ShapeNet Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval-p2m", action="store_true", help="pix2mesh evaluation mode")
    parser.add_argument("--no-color", action="store_true", help="disable colorful logging")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus per machine")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "--data-dir",
        default="./datasets/shapenet/ShapeNetV1processed.zip",
        help="Path to the ShapeNet zipped data from preprocessing - used ONLY when copying data",
    )
    parser.add_argument("--tmp-dir", default="/tmp")
    parser.add_argument("--copy-data", action="store_true", help="copy data")
    parser.add_argument(
        "--torch-home", default="$XDG_CACHE_HOME/torch", help="Path to torchvision model zoo"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()
