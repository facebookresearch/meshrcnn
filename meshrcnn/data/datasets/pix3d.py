# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import io
import logging
import os
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_pix3d_json"]


def load_pix3d_json(json_file, image_root, dataset_name=None):
    """
    Load a json file with Pix3D's instances annotation format.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.

    Returns:
        list[dict]: a list of dicts in "Detectron2 Dataset" format. (See DATASETS.md)

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    assert dataset_name is not None
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        logger.warning(
            """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
        )
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(list(coco_api.imgs.keys()))

    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:

        # examples with imgfiles = {img/table/1749.jpg, img/table/0045.png}
        # have a mismatch between images and masks. Thus, ignore
        if img_dict["file_name"] in ["img/table/1749.jpg", "img/table/0045.png"]:
            continue

        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0

            obj = {
                field: anno[field] for field in ["iscrowd", "bbox", "category_id"] if field in anno
            }

            segm = anno.get("segmentation", None)
            if segm:  # string
                obj["segmentation"] = os.path.join(image_root, segm)

            voxel = anno.get("voxel", None)
            if voxel:
                obj["voxel"] = os.path.join(image_root, voxel)

            mesh = anno.get("model", None)
            if mesh:
                obj["mesh"] = mesh

            # camera
            obj["K"] = anno["K"]
            obj["R"] = anno["rot_mat"]
            obj["t"] = anno["trans_mat"]

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    """
    Test the Pix3D json dataset loader.

    Usage:
        python -m meshrcnn.data.datasets.pix3d \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco", "coco_person", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from meshrcnn.utils.vis import draw_pix3d_dict
    import cv2
    import sys

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_pix3d_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "pix3d-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        vis = draw_pix3d_dict(d, meta.thing_classes + ["0"])
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        cv2.imwrite(fpath, vis)
