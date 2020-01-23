#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from setuptools import find_packages, setup

setup(
    name="meshrcnn",
    version="1.0",
    author="FAIR",
    url="https://github.com/facebookresearch/meshrcnn",
    description="Code for Mesh R-CNN",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["torchvision>=0.4", "fvcore", "detectron2", "pytorch3d"],
)
