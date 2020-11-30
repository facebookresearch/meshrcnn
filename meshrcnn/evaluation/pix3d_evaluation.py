# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import detectron2.utils.comm as comm
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import meshrcnn.utils.VOCap as VOCap
from meshrcnn.utils import shape as shape_utils
from meshrcnn.utils import vis as vis_utils
from meshrcnn.utils.metrics import compare_meshes

logger = logging.getLogger(__name__)


class Pix3DEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and meshes
    outputs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._device = cfg.MODEL.DEVICE
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._filter_iou = 0.3

        # load unique obj files
        assert dataset_name is not None
        # load unique obj meshes
        # Pix3D models are few in number (= 735) thus it's more efficient
        # to load them in memory rather than read them at every iteration
        logger.info("Loading unique objects from {}...".format(dataset_name))
        json_file = MetadataCatalog.get(dataset_name).json_file
        model_root = MetadataCatalog.get(dataset_name).image_root
        self._mesh_models = load_unique_meshes(json_file, model_root)
        logger.info("Unique objects loaded: {}".format(len(self._mesh_models)))

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.MESH_ON or cfg.MODEL.VOXEL_ON:
            tasks = tasks + ("mesh",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)

                if instances.has("pred_masks"):
                    # use RLE to encode the masks, because they are too large and takes memory
                    # since this evaluator stores outputs of the entire dataset
                    rles = [
                        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                        for mask in instances.pred_masks
                    ]
                    for rle in rles:
                        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                        # json writer which always produces strings cannot serialize a bytestream
                        # unless you decode it. Thankfully, utf-8 works out (which is also what
                        # the pycocotools/_mask.pyx does).
                        rle["counts"] = rle["counts"].decode("utf-8")
                    instances.pred_masks_rle = rles
                    instances.remove("pred_masks")
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return

        if self._output_dir:
            torch.save(
                self._predictions, os.path.join(self._output_dir, "instances_predictions.pth")
            )

        self._results = OrderedDict()

        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate mesh rcnn predictions.
        """

        if "segm" in self._tasks and "mesh" in self._tasks:
            results = evaluate_for_pix3d(
                self._predictions,
                self._coco_api,
                self._metadata,
                self._filter_iou,
                mesh_models=self._mesh_models,
                device=self._device,
            )

            # print results
            self._logger.info("Box AP %.5f" % (results["box_ap@%.1f" % 0.5]))
            self._logger.info("Mask AP %.5f" % (results["mask_ap@%.1f" % 0.5]))
            self._logger.info("Mesh AP %.5f" % (results["mesh_ap@%.1f" % 0.5]))
            self._results["shape"] = results


def evaluate_for_pix3d(
    predictions,
    dataset,
    metadata,
    filter_iou,
    mesh_models=None,
    iou_thresh=0.5,
    mask_thresh=0.5,
    device=None,
    vis_preds=False,
):
    from PIL import Image

    if device is None:
        device = torch.device("cpu")

    F1_TARGET = "F1@0.300000"

    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    mask_apscores, mask_aplabels = {}, {}
    mesh_apscores, mesh_aplabels = {}, {}
    npos = {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mask_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mask_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mesh_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mesh_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        npos[cat_id] = 0.0
    box_covered = []
    mask_covered = []
    mesh_covered = []

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        # examples with imgfiles = {img/table/1749.jpg, img/table/0045.png}
        # have a mismatch between images and masks. Thus, ignore
        image_file_name = dataset.loadImgs([gt_ann["image_id"]])[0]["file_name"]
        if image_file_name in ["img/table/1749.jpg", "img/table/0045.png"]:
            continue
        npos[gt_label] += 1.0

    for prediction in predictions:

        original_id = prediction["image_id"]
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        image_size = [image_height, image_width]
        image_file_name = dataset.loadImgs([original_id])[0]["file_name"]
        # examples with imgfiles = {img/table/1749.jpg, img/table/0045.png}
        # have a mismatch between images and masks. Thus, ignore
        if image_file_name in ["img/table/1749.jpg", "img/table/0045.png"]:
            continue

        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        # predictions
        scores = prediction["instances"].scores
        boxes = prediction["instances"].pred_boxes.to(device)
        labels = prediction["instances"].pred_classes
        masks_rles = prediction["instances"].pred_masks_rle
        if hasattr(prediction["instances"], "pred_meshes"):
            meshes = prediction["instances"].pred_meshes  # preditected meshes
            verts = [mesh[0] for mesh in meshes]
            faces = [mesh[1] for mesh in meshes]
            meshes = Meshes(verts=verts, faces=faces).to(device)
        else:
            meshes = ico_sphere(4, device)
            meshes = meshes.extend(num_img_preds).to(device)
        if hasattr(prediction["instances"], "pred_dz"):
            pred_dz = prediction["instances"].pred_dz
            heights = boxes.tensor[:, 3] - boxes.tensor[:, 1]
            # NOTE see appendix for derivation of pred dz
            pred_dz = pred_dz[:, 0] * heights.cpu()
        else:
            raise ValueError("Z range of box not predicted")
        assert prediction["instances"].image_size[0] == image_height
        assert prediction["instances"].image_size[1] == image_width

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        assert len(gt_ann_ids) == 1  # note that pix3d has one annotation per image
        gt_anns = dataset.loadAnns(gt_ann_ids)[0]
        assert gt_anns["image_id"] == original_id

        # get original ground truth mask, box, label & mesh
        maskfile = os.path.join(metadata.image_root, gt_anns["segmentation"])
        with PathManager.open(maskfile, "rb") as f:
            gt_mask = torch.tensor(np.asarray(Image.open(f), dtype=np.float32) / 255.0)
        assert gt_mask.shape[0] == image_height and gt_mask.shape[1] == image_width

        gt_mask = (gt_mask > 0).to(dtype=torch.uint8)  # binarize mask
        gt_mask_rle = [mask_util.encode(np.array(gt_mask[:, :, None], order="F"))[0]]
        gt_box = np.array(gt_anns["bbox"]).reshape(-1, 4)  # xywh from coco
        gt_box = BoxMode.convert(gt_box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        gt_label = gt_anns["category_id"]
        faux_gt_targets = Boxes(torch.tensor(gt_box, dtype=torch.float32, device=device))

        # load gt mesh and extrinsics/intrinsics
        gt_R = torch.tensor(gt_anns["rot_mat"]).to(device)
        gt_t = torch.tensor(gt_anns["trans_mat"]).to(device)
        gt_K = torch.tensor(gt_anns["K"]).to(device)
        if mesh_models is not None:
            modeltype = gt_anns["model"]
            gt_verts, gt_faces = (
                mesh_models[modeltype][0].clone(),
                mesh_models[modeltype][1].clone(),
            )
            gt_verts = gt_verts.to(device)
            gt_faces = gt_faces.to(device)
        else:
            # load from disc
            raise NotImplementedError
        gt_verts = shape_utils.transform_verts(gt_verts, gt_R, gt_t)
        gt_zrange = torch.stack([gt_verts[:, 2].min(), gt_verts[:, 2].max()])
        gt_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])

        # box iou
        boxiou = pairwise_iou(boxes, faux_gt_targets)

        # filter predictions with iou > filter_iou
        valid_pred_ids = boxiou > filter_iou

        # mask iou
        miou = mask_util.iou(masks_rles, gt_mask_rle, [0])

        # # gt zrange (zrange stores min_z and max_z)
        # # zranges = torch.stack([gt_zrange] * len(meshes), dim=0)

        # predicted zrange (= pred_dz)
        assert hasattr(prediction["instances"], "pred_dz")
        # It's impossible to predict the center location in Z (=tc)
        # from the image. See appendix for more.
        tc = (gt_zrange[1] + gt_zrange[0]) / 2.0
        # Given a center location (tc) and a focal_length,
        # pred_dz = pred_dz * box_h * tc / focal_length
        # See appendix for more.
        zranges = torch.stack(
            [
                torch.stack(
                    [tc - tc * pred_dz[i] / 2.0 / gt_K[0], tc + tc * pred_dz[i] / 2.0 / gt_K[0]]
                )
                for i in range(len(meshes))
            ],
            dim=0,
        )

        gt_Ks = gt_K.view(1, 3).expand(len(meshes), 3)
        meshes = transform_meshes_to_camera_coord_system(
            meshes, boxes.tensor, zranges, gt_Ks, image_size
        )

        if vis_preds:
            vis_utils.visualize_predictions(
                original_id,
                image_file_name,
                scores,
                labels,
                boxes.tensor,
                masks_rles,
                meshes,
                metadata,
                "/tmp/output",
            )

        shape_metrics = compare_meshes(meshes, gt_mesh, reduce=False)

        # sort predictions in descending order
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)

        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction
            if valid_pred_ids[idx_sorted[pred_id], 0] == 0:
                continue
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]].item()]
            pred_miou = miou[idx_sorted[pred_id]].item()
            pred_biou = boxiou[idx_sorted[pred_id]].item()
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)
            # note that metrics returns f1 in % (=x100)
            pred_f1 = shape_metrics[F1_TARGET][idx_sorted[pred_id]].item() / 100.0

            # mask
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (original_id not in mask_covered)
            ):
                tpfp[0] = 1
                mask_covered.append(original_id)
            mask_apscores[pred_label].append(pred_score)
            mask_aplabels[pred_label].append(tpfp)

            # box
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (original_id not in box_covered)
            ):
                tpfp[0] = 1
                box_covered.append(original_id)
            box_apscores[pred_label].append(pred_score)
            box_aplabels[pred_label].append(tpfp)

            # mesh
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_f1 > iou_thresh)
                and (original_id not in mesh_covered)
            ):
                tpfp[0] = 1
                mesh_covered.append(original_id)
            mesh_apscores[pred_label].append(pred_score)
            mesh_aplabels[pred_label].append(tpfp)

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    pix3d_metrics = {}
    boxap, maskap, meshap = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        cat_box_ap = VOCap.compute_ap(
            torch.cat(box_apscores[cat_id]), torch.cat(box_aplabels[cat_id]), npos[cat_id]
        )
        boxap += cat_box_ap
        pix3d_metrics["box_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_box_ap

        cat_mask_ap = VOCap.compute_ap(
            torch.cat(mask_apscores[cat_id]), torch.cat(mask_aplabels[cat_id]), npos[cat_id]
        )
        maskap += cat_mask_ap
        pix3d_metrics["mask_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_mask_ap

        cat_mesh_ap = VOCap.compute_ap(
            torch.cat(mesh_apscores[cat_id]), torch.cat(mesh_aplabels[cat_id]), npos[cat_id]
        )
        meshap += cat_mesh_ap
        pix3d_metrics["mesh_ap@%.1f - %s" % (iou_thresh, cat_name)] = cat_mesh_ap

    pix3d_metrics["box_ap@%.1f" % iou_thresh] = boxap / valid
    pix3d_metrics["mask_ap@%.1f" % iou_thresh] = maskap / valid
    pix3d_metrics["mesh_ap@%.1f" % iou_thresh] = meshap / valid

    # print test ground truth
    vis_utils.print_instances_class_histogram(
        [npos[cat_id] for cat_id in cat_ids],  # number of instances
        [dataset.loadCats([cat_id])[0]["name"] for cat_id in cat_ids],  # class names
        pix3d_metrics,
    )

    return pix3d_metrics


def transform_meshes_to_camera_coord_system(meshes, boxes, zranges, Ks, imsize):
    device = meshes.device
    new_verts, new_faces = [], []
    h, w = imsize
    im_size = torch.tensor([w, h], device=device).view(1, 2)
    assert len(meshes) == len(zranges)
    for i in range(len(meshes)):
        verts, faces = meshes.get_mesh_verts_faces(i)
        if verts.numel() == 0:
            verts, faces = ico_sphere(level=3, device=device).get_mesh_verts_faces(0)
        assert not torch.isnan(verts).any()
        assert not torch.isnan(faces).any()
        roi = boxes[i].view(1, 4)
        zrange = zranges[i].view(1, 2)
        K = Ks[i].view(1, 3)
        cub3D = shape_utils.box2D_to_cuboid3D(zrange, K, roi, im_size)
        txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)

        # image to camera coords
        verts[:, 0] = -verts[:, 0]
        verts[:, 1] = -verts[:, 1]

        # transform to destination size
        xz = verts[:, [0, 2]]
        yz = verts[:, [1, 2]]
        pxz = txz.inverse(xz.view(1, -1, 2)).squeeze(0)
        pyz = tyz.inverse(yz.view(1, -1, 2)).squeeze(0)
        verts = torch.stack([pxz[:, 0], pyz[:, 0], pxz[:, 1]], dim=1).to(
            device, dtype=torch.float32
        )

        new_verts.append(verts)
        new_faces.append(faces)

    return Meshes(verts=new_verts, faces=new_faces)


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
