# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from collections import defaultdict
import detectron2.utils.comm as comm
import torch
from detectron2.evaluation import inference_context

from meshrcnn.utils.metrics import compare_meshes

import shapenet.utils.vis as vis_utils
from shapenet.data.utils import image_to_numpy, imagenet_deprocess

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_test(model, data_loader, vis_preds=False):
    """
    This function evaluates the model on the dataset defined by data_loader.
    The metrics reported are described in Table 2 of our paper.
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    deprocess = imagenet_deprocess(rescale_image=False)
    device = torch.device("cuda:0")
    # evaluation
    class_names = {
        "02828884": "bench",
        "03001627": "chair",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "firearm",
        "04379243": "table",
        "04530566": "watercraft",
        "02691156": "plane",
        "02933112": "cabinet",
        "02958343": "car",
        "03211117": "monitor",
        "04256520": "couch",
        "04401088": "cellphone",
    }

    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    normal = {i: 0 for i in class_names}
    f1_01 = {i: 0 for i in class_names}
    f1_03 = {i: 0 for i in class_names}
    f1_05 = {i: 0 for i in class_names}

    num_batch_evaluated = 0
    for batch in data_loader:
        batch = data_loader.postprocess(batch, device)
        imgs, meshes_gt, _, _, _, id_strs = batch
        sids = [id_str.split("-")[0] for id_str in id_strs]
        for sid in sids:
            num_instances[sid] += 1

        with inference_context(model):
            voxel_scores, meshes_pred = model(imgs)
            cur_metrics = compare_meshes(meshes_pred[-1], meshes_gt, reduce=False)
            cur_metrics["verts_per_mesh"] = meshes_pred[-1].num_verts_per_mesh().cpu()
            cur_metrics["faces_per_mesh"] = meshes_pred[-1].num_faces_per_mesh().cpu()

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
                normal[sid] += cur_metrics["AbsNormalConsistency"][i].item()
                f1_01[sid] += cur_metrics["F1@%f" % 0.1][i].item()
                f1_03[sid] += cur_metrics["F1@%f" % 0.3][i].item()
                f1_05[sid] += cur_metrics["F1@%f" % 0.5][i].item()

                if vis_preds:
                    img = image_to_numpy(deprocess(imgs[i]))
                    vis_utils.visualize_prediction(
                        id_strs[i], img, meshes_pred[-1][i], "/tmp/output"
                    )

            num_batch_evaluated += 1
            logger.info("Evaluated %d / %d batches" % (num_batch_evaluated, len(data_loader)))

    vis_utils.print_instances_class_histogram(
        num_instances,
        class_names,
        {"chamfer": chamfer, "normal": normal, "f1_01": f1_01, "f1_03": f1_03, "f1_05": f1_05},
    )


@torch.no_grad()
def evaluate_test_p2m(model, data_loader):
    """
    This function evaluates the model on the dataset defined by data_loader.
    The metrics reported are described in Table 1 of our paper, following previous
    reported approaches (like Pixel2Mesh - p2m), where meshes are
    rescaled by a factor of 0.57. See the paper for more details.
    """
    assert comm.is_main_process()
    device = torch.device("cuda:0")
    # evaluation
    class_names = {
        "02828884": "bench",
        "03001627": "chair",
        "03636649": "lamp",
        "03691459": "speaker",
        "04090263": "firearm",
        "04379243": "table",
        "04530566": "watercraft",
        "02691156": "plane",
        "02933112": "cabinet",
        "02958343": "car",
        "03211117": "monitor",
        "04256520": "couch",
        "04401088": "cellphone",
    }

    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    normal = {i: 0 for i in class_names}
    f1_1e_4 = {i: 0 for i in class_names}
    f1_2e_4 = {i: 0 for i in class_names}

    num_batch_evaluated = 0
    for batch in data_loader:
        batch = data_loader.postprocess(batch, device)
        imgs, meshes_gt, _, _, _, id_strs = batch
        sids = [id_str.split("-")[0] for id_str in id_strs]
        for sid in sids:
            num_instances[sid] += 1

        with inference_context(model):
            voxel_scores, meshes_pred = model(imgs)
            # NOTE that for the F1 thresholds we take the square root of 1e-4 & 2e-4
            # as `compare_meshes` returns the euclidean distance (L2) of two pointclouds.
            # In Pixel2Mesh, the squared L2 (L2^2) is computed instead.
            # i.e. (L2^2 < τ) <=> (L2 < sqrt(τ))
            cur_metrics = compare_meshes(
                meshes_pred[-1], meshes_gt, scale=0.57, thresholds=[0.01, 0.014142], reduce=False
            )
            cur_metrics["verts_per_mesh"] = meshes_pred[-1].num_verts_per_mesh().cpu()
            cur_metrics["faces_per_mesh"] = meshes_pred[-1].num_faces_per_mesh().cpu()

            for i, sid in enumerate(sids):
                chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
                normal[sid] += cur_metrics["AbsNormalConsistency"][i].item()
                f1_1e_4[sid] += cur_metrics["F1@%f" % 0.01][i].item()
                f1_2e_4[sid] += cur_metrics["F1@%f" % 0.014142][i].item()

            num_batch_evaluated += 1
            logger.info("Evaluated %d / %d batches" % (num_batch_evaluated, len(data_loader)))

    vis_utils.print_instances_class_histogram_p2m(
        num_instances,
        class_names,
        {"chamfer": chamfer, "normal": normal, "f1_1e_4": f1_1e_4, "f1_2e_4": f1_2e_4},
    )


@torch.no_grad()
def evaluate_split(
    model, loader, max_predictions=-1, num_predictions_keep=10, prefix="", store_predictions=False
):
    """
    This function is used to report validation performance during training.
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    device = torch.device("cuda:0")
    num_predictions = 0
    num_predictions_kept = 0
    predictions = defaultdict(list)
    metrics = defaultdict(list)
    deprocess = imagenet_deprocess(rescale_image=False)
    for batch in loader:
        batch = loader.postprocess(batch, device)
        imgs, meshes_gt, points_gt, normals_gt, voxels_gt = batch
        voxel_scores, meshes_pred = model(imgs)

        # Only compute metrics for the final predicted meshes, not intermediates
        cur_metrics = compare_meshes(meshes_pred[-1], meshes_gt)
        if cur_metrics is None:
            continue
        for k, v in cur_metrics.items():
            metrics[k].append(v)

        # Store input images and predicted meshes
        if store_predictions:
            N = imgs.shape[0]
            for i in range(N):
                if num_predictions_kept >= num_predictions_keep:
                    break
                num_predictions_kept += 1

                img = image_to_numpy(deprocess(imgs[i]))
                predictions["%simg_input" % prefix].append(img)
                for level, cur_meshes_pred in enumerate(meshes_pred):
                    verts, faces = cur_meshes_pred.get_mesh(i)
                    verts_key = "%sverts_pred_%d" % (prefix, level)
                    faces_key = "%sfaces_pred_%d" % (prefix, level)
                    predictions[verts_key].append(verts.cpu().numpy())
                    predictions[faces_key].append(faces.cpu().numpy())

        num_predictions += len(meshes_gt)
        logger.info("Evaluated %d predictions so far" % num_predictions)
        if 0 < max_predictions <= num_predictions:
            break

    # Average numeric metrics, and concatenate images
    metrics = {"%s%s" % (prefix, k): np.mean(v) for k, v in metrics.items()}
    if store_predictions:
        img_key = "%simg_input" % prefix
        predictions[img_key] = np.stack(predictions[img_key], axis=0)

    return metrics, predictions
