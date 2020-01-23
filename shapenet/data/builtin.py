# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths
"""
import os

# each dataset contains name : (data_dir, splits_file)
_PREDEFINED_SPLITS_SHAPENET = {
    "shapenet": ("shapenet/ShapeNetV1processed", "shapenet/pix2mesh_splits_val05.json")
}


def register_shapenet(dataset_name, root="datasets"):
    if dataset_name not in _PREDEFINED_SPLITS_SHAPENET.keys():
        raise ValueError("%s not registered" % dataset_name)
    data_dir = _PREDEFINED_SPLITS_SHAPENET[dataset_name][0]
    splits_file = _PREDEFINED_SPLITS_SHAPENET[dataset_name][1]
    return os.path.join(root, data_dir), os.path.join(root, splits_file)
