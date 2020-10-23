# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import json
import logging
import numpy as np
import os
import torch
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances
from fvcore.common.file_io import PathManager
from pytorch3d.io import load_obj

from meshrcnn.structures import MeshInstances, VoxelInstances
from meshrcnn.utils import shape as shape_utils

from PIL import Image

__all__ = ["MeshRCNNMapper"]

logger = logging.getLogger(__name__)


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width

    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        masks = [obj["segmentation"] for obj in annos]
        target.gt_masks = torch.stack(masks, dim=0)

    # camera
    if len(annos) and "K" in annos[0]:
        K = [torch.tensor(obj["K"]) for obj in annos]
        target.gt_K = torch.stack(K, dim=0)

    if len(annos) and "voxel" in annos[0]:
        voxels = [obj["voxel"] for obj in annos]
        target.gt_voxels = VoxelInstances(voxels)

    if len(annos) and "mesh" in annos[0]:
        meshes = [obj["mesh"] for obj in annos]
        target.gt_meshes = MeshInstances(meshes)

    if len(annos) and "dz" in annos[0]:
        dz = [obj["dz"] for obj in annos]
        target.gt_dz = torch.tensor(dz)

    return target


class MeshRCNNMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.voxel_on       = cfg.MODEL.VOXEL_ON
        self.mesh_on        = cfg.MODEL.MESH_ON
        self.zpred_on       = cfg.MODEL.ZPRED_ON
        # fmt: on

        if cfg.MODEL.LOAD_PROPOSALS:
            raise ValueError("Loading pre-computed proposals is not supported.")

        self.is_train = is_train

        assert dataset_names is not None
        # load unique obj meshes
        # Pix3D models are few in number (= 735) thus it's more efficient
        # to load them in memory rather than read them at every iteration
        all_mesh_models = {}
        for dataset_name in dataset_names:
            json_file = MetadataCatalog.get(dataset_name).json_file
            model_root = MetadataCatalog.get(dataset_name).image_root
            logger.info("Loading models from {}...".format(dataset_name))
            dataset_mesh_models = load_unique_meshes(json_file, model_root)
            all_mesh_models.update(dataset_mesh_models)
            logger.info("Unique objects loaded: {}".format(len(dataset_mesh_models)))

        self._all_mesh_models = all_mesh_models

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """
        # get 3D models for each annotations and remove 3D mesh models from image dict
        mesh_models = []
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                mesh_models.append(
                    [
                        self._all_mesh_models[anno["mesh"]][0].clone(),
                        self._all_mesh_models[anno["mesh"]][1].clone(),
                    ]
                )

        dataset_dict = {key: value for key, value in dataset_dict.items() if key != "mesh_models"}
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if "annotations" in dataset_dict:
            for i, anno in enumerate(dataset_dict["annotations"]):
                anno["mesh"] = mesh_models[i]

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                self.transform_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Should not be empty during training
            instances = annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def transform_annotations(self, annotation, transforms, image_size):
        """
        Apply image transformations to the annotations.

        After this method, the box mode will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        # each instance contains 1 mask
        if self.mask_on and "segmentation" in annotation:
            annotation["segmentation"] = self._process_mask(annotation["segmentation"], transforms)
        else:
            annotation.pop("segmentation", None)

        # camera
        h, w = image_size
        annotation["K"] = [annotation["K"][0], w / 2.0, h / 2.0]
        annotation["R"] = torch.tensor(annotation["R"])
        annotation["t"] = torch.tensor(annotation["t"])

        if self.zpred_on and "mesh" in annotation:
            annotation["dz"] = self._process_dz(
                annotation["mesh"],
                transforms,
                focal_length=annotation["K"][0],
                R=annotation["R"],
                t=annotation["t"],
            )
        else:
            annotation.pop("dz", None)

        # each instance contains 1 voxel
        if self.voxel_on and "voxel" in annotation:
            annotation["voxel"] = self._process_voxel(
                annotation["voxel"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("voxel", None)

        # each instance contains 1 mesh
        if self.mesh_on and "mesh" in annotation:
            annotation["mesh"] = self._process_mesh(
                annotation["mesh"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("mesh", None)

        return annotation

    def _process_dz(self, mesh, transforms, focal_length=1.0, R=None, t=None):
        # clone mesh
        verts, faces = mesh
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)
        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        dz = verts[:, 2].max() - verts[:, 2].min()
        z_center = (verts[:, 2].max() + verts[:, 2].min()) / 2.0
        dz = dz / z_center
        dz = dz * focal_length
        for t in transforms.transforms:
            # NOTE normalize the dz by the height scaling of the image.
            # This is necessary s.t. z-regression targets log(dz/roi_h)
            # are invariant to the scaling of the roi_h
            if isinstance(t, T.ResizeTransform):
                dz = dz * (t.new_h * 1.0 / t.h)
        return dz

    def _process_mask(self, mask, transforms):
        # applies image transformations to mask
        with PathManager.open(mask, "rb") as f:
            mask = np.asarray(Image.open(f))
        mask = transforms.apply_image(mask)
        mask = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32) / 255.0
        return mask

    def _process_voxel(self, voxel, transforms, R=None, t=None):
        # read voxel
        verts = shape_utils.read_voxel(voxel)
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)

        # applies image transformations to voxels (represented as verts)
        # NOTE this function does not support generic transforms in T
        # the apply_coords functionality works for voxels for the following
        # transforms (HFlipTransform, NoOpTransform, ResizeTransform)
        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        for t in transforms.transforms:
            if isinstance(t, T.HFlipTransform):
                verts[:, 0] = -verts[:, 0]
            elif isinstance(t, T.ResizeTransform):
                verts = t.apply_coords(verts)
            elif isinstance(t, T.NoOpTransform):
                pass
            else:
                raise ValueError("Transform {} not recognized".format(t))
        return verts

    def _process_mesh(self, mesh, transforms, R=None, t=None):
        # clone mesh
        verts, faces = mesh
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)

        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        for t in transforms.transforms:
            if isinstance(t, T.HFlipTransform):
                verts[:, 0] = -verts[:, 0]
            elif isinstance(t, T.ResizeTransform):
                verts = t.apply_coords(verts)
            elif isinstance(t, T.NoOpTransform):
                pass
            else:
                raise ValueError("Transform {} not recognized".format(t))
        return verts, faces


def load_unique_meshes(json_file, model_root):
    with PathManager.open(json_file, "r") as f:
        anns = json.load(f)["annotations"]
    # find unique models
    unique_models = []
    for obj in anns:
        model_type = obj["model"]
        if model_type not in unique_models:
            unique_models.append(model_type)
    # read unique models
    object_models = {}
    for model in unique_models:
        with PathManager.open(os.path.join(model_root, model), "rb") as f:
            mesh = load_obj(f, load_textures=False)
        object_models[model] = [mesh[0], mesh[1].verts_idx]
    return object_models
