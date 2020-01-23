# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.file_io import PathHandler, PathManager

__all__ = ["ShapenetHandler"]


class ShapenetHandler(PathHandler):
    """
    Resolve anything that's in Mesh R-CNN's model zoo.
    """

    PREFIX = "shapenet://"
    SHAPENET_PREFIX = "https://dl.fbaipublicfiles.com/meshrcnn/shapenet/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.SHAPENET_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(ShapenetHandler())
