#!/bin/bash -e
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Download R2N2 and associated splits
# User needs to register and download ShapeNetCore.v1 and ShapeNetCore.v1.binvox

cd "${0%/*}"

# download r2n2 renderings
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xvzf ShapeNetRendering.tgz

# downloand splits
BASE=https://dl.fbaipublicfiles.com/meshrcnn
wget $BASE/shapenet/pix2mesh_splits_val05.json
