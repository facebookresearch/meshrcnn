#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Download R2N2 and associated splits
# Note that ShapeNet Core v1 should exist in the same directory

cd "${0%/*}"

wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xvzf ShapeNetRendering.tgz

BASE=https://dl.fbaipublicfiles.com/meshrcnn

wget $BASE/shapenet/pix2mesh_splits_val05.json
