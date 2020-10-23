#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import shutil
import time
import detectron2.utils.comm as comm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager

from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet
from shapenet.evaluation import evaluate_split, evaluate_test, evaluate_test_p2m

# required so that .register() calls are executed in module scope
from shapenet.modeling import MeshLoss, build_model
from shapenet.solver import build_lr_scheduler, build_optimizer
from shapenet.utils import Checkpoint, Timer, clean_state_dict, default_argument_parser

logger = logging.getLogger("shapenet")


def copy_data(args):
    data_base, data_ext = os.path.splitext(os.path.basename(args.data_dir))
    assert data_ext in [".tar", ".zip"]
    t0 = time.time()
    # Copy file from source to a local file
    logger.info("Copying %s to %s ..." % (args.data_dir, args.tmp_dir))
    zip_local_path = PathManager.get_local_path(args.data_dir)
    t1 = time.time()
    logger.info("Copying took %fs" % (t1 - t0))
    shutil.unpack_archive(zip_local_path, args.tmp_dir)
    logger.info("Unpacking %s ..." % zip_local_path)
    t2 = time.time()
    logger.info("Unpacking took %f" % (t2 - t1))
    args.data_dir = os.path.join(args.tmp_dir, data_base)
    logger.info("args.data_dir = %s" % args.data_dir)


def main_worker_eval(worker_id, args):

    device = torch.device("cuda:%d" % worker_id)
    cfg = setup(args)

    # build test set
    test_loader = build_data_loader(cfg, "MeshVox", "test", multigpu=False)
    logger.info("test - %d" % len(test_loader))

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))
    state_dict = clean_state_dict(cp["best_states"]["model"])
    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    if args.eval_p2m:
        evaluate_test_p2m(model, test_loader)
    else:
        evaluate_test(model, test_loader)


def main_worker(worker_id, args):
    distributed = False
    if args.num_gpus > 1:
        distributed = True
        dist.init_process_group(
            backend="NCCL", init_method=args.dist_url, world_size=args.num_gpus, rank=worker_id
        )
        torch.cuda.set_device(worker_id)

    device = torch.device("cuda:%d" % worker_id)

    cfg = setup(args)

    # data loaders
    loaders = setup_loaders(cfg)
    for split_name, loader in loaders.items():
        logger.info("%s - %d" % (split_name, len(loader)))

    # build the model
    model = build_model(cfg)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[worker_id],
            output_device=worker_id,
            check_reduction=True,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(cfg, model)
    cfg.SOLVER.COMPUTED_MAX_ITERS = cfg.SOLVER.NUM_EPOCHS * len(loaders["train"])
    scheduler = build_lr_scheduler(cfg, optimizer)

    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "voxel_weight": cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
    }
    loss_fn = MeshLoss(**loss_fn_kwargs)

    checkpoint_path = "checkpoint.pt"
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_path)
    cp = Checkpoint(checkpoint_path)
    if len(cp.restarts) == 0:
        # We are starting from scratch, so store some initial data in cp
        iter_per_epoch = len(loaders["train"])
        cp.store_data("iter_per_epoch", iter_per_epoch)
    else:
        logger.info("Loading model state from checkpoint")
        model.load_state_dict(cp.latest_states["model"])
        optimizer.load_state_dict(cp.latest_states["optim"])
        scheduler.load_state_dict(cp.latest_states["lr_scheduler"])

    training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn)


def training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn):
    Timer.timing = False
    iteration_timer = Timer("Iteration")

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    if hasattr(model, "module"):
        params = list(model.module.parameters())
    else:
        params = list(model.parameters())
    loss_moving_average = cp.data.get("loss_moving_average", None)
    while cp.epoch < cfg.SOLVER.NUM_EPOCHS:
        if comm.is_main_process():
            logger.info("Starting epoch %d / %d" % (cp.epoch + 1, cfg.SOLVER.NUM_EPOCHS))

        # When using a DistributedSampler we need to manually set the epoch so that
        # the data is shuffled differently at each epoch
        for loader in loaders.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(cp.epoch)

        for i, batch in enumerate(loaders["train"]):
            if i == 0:
                iteration_timer.start()
            else:
                iteration_timer.tick()
            batch = loaders["train"].postprocess(batch, device)
            imgs, meshes_gt, points_gt, normals_gt, voxels_gt = batch

            num_infinite_params = 0
            for p in params:
                num_infinite_params += (torch.isfinite(p.data) == 0).sum().item()
            if num_infinite_params > 0:
                msg = "ERROR: Model has %d non-finite params (before forward!)"
                logger.info(msg % num_infinite_params)
                return

            model_kwargs = {}
            if cfg.MODEL.VOXEL_ON and cp.t < cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS:
                model_kwargs["voxel_only"] = True
            with Timer("Forward"):
                voxel_scores, meshes_pred = model(imgs, **model_kwargs)

            num_infinite = 0
            for cur_meshes in meshes_pred:
                cur_verts = cur_meshes.verts_packed()
                num_infinite += (torch.isfinite(cur_verts) == 0).sum().item()
            if num_infinite > 0:
                logger.info("ERROR: Got %d non-finite verts" % num_infinite)
                return

            loss, losses = None, {}
            if num_infinite == 0:
                loss, losses = loss_fn(
                    voxel_scores, meshes_pred, voxels_gt, (points_gt, normals_gt)
                )
            skip = loss is None
            if loss is None or (torch.isfinite(loss) == 0).sum().item() > 0:
                logger.info("WARNING: Got non-finite loss %f" % loss)
                skip = True

            if model_kwargs.get("voxel_only", False):
                for k, v in losses.items():
                    if k != "voxel":
                        losses[k] = 0.0 * v

            if loss is not None and cp.t % cfg.SOLVER.LOGGING_PERIOD == 0:
                if comm.is_main_process():
                    cp.store_metric(loss=loss.item())
                    str_out = "Iteration: %d, epoch: %d, lr: %.5f," % (
                        cp.t,
                        cp.epoch,
                        optimizer.param_groups[0]["lr"],
                    )
                    for k, v in losses.items():
                        str_out += "  %s loss: %.4f," % (k, v.item())
                    str_out += "  total loss: %.4f," % loss.item()

                    # memory allocaged
                    if torch.cuda.is_available():
                        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                        str_out += " mem: %d" % max_mem_mb

                    if len(meshes_pred) > 0:
                        mean_V = meshes_pred[-1].num_verts_per_mesh().float().mean().item()
                        mean_F = meshes_pred[-1].num_faces_per_mesh().float().mean().item()
                        str_out += ", mesh size = (%d, %d)" % (mean_V, mean_F)
                    logger.info(str_out)

            if loss_moving_average is None and loss is not None:
                loss_moving_average = loss.item()

            # Skip backprop for this batch if the loss is above the skip factor times
            # the moving average for losses
            if loss is None:
                pass
            elif loss.item() > cfg.SOLVER.SKIP_LOSS_THRESH * loss_moving_average:
                logger.info("Warning: Skipping loss %f on GPU %d" % (loss.item(), comm.get_rank()))
                cp.store_metric(losses_skipped=loss.item())
                skip = True
            else:
                # Update the moving average of our loss
                gamma = cfg.SOLVER.LOSS_SKIP_GAMMA
                loss_moving_average *= gamma
                loss_moving_average += (1.0 - gamma) * loss.item()
                cp.store_data("loss_moving_average", loss_moving_average)

            if skip:
                logger.info("Dummy backprop on GPU %d" % comm.get_rank())
                loss = 0.0 * sum(p.sum() for p in params)

            # Backprop and step
            scheduler.step()
            optimizer.zero_grad()
            with Timer("Backward"):
                loss.backward()

            # When training with normal loss, sometimes I get NaNs in gradient that
            # cause the model to explode. Check for this before performing a gradient
            # update. This is safe in mult-GPU since gradients have already been
            # summed, so each GPU has the same gradients.
            num_infinite_grad = 0
            for p in params:
                num_infinite_grad += (torch.isfinite(p.grad) == 0).sum().item()
            if num_infinite_grad == 0:
                optimizer.step()
            else:
                msg = "WARNING: Got %d non-finite elements in gradient; skipping update"
                logger.info(msg % num_infinite_grad)
            cp.step()

            if cp.t % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                eval_and_save(model, loaders, optimizer, scheduler, cp)
        cp.step_epoch()
    eval_and_save(model, loaders, optimizer, scheduler, cp)

    if comm.is_main_process():
        logger.info("Evaluating on test set:")
        test_loader = build_data_loader(cfg, "MeshVox", "test", multigpu=False)
        evaluate_test(model, test_loader)


def eval_and_save(model, loaders, optimizer, scheduler, cp):
    # NOTE(gkioxari) For now only do evaluation on the main process
    if comm.is_main_process():
        logger.info("Evaluating on training set:")
        train_metrics, train_preds = evaluate_split(
            model, loaders["train_eval"], prefix="train_", max_predictions=1000
        )
        eval_split = "val"
        if eval_split not in loaders:
            logger.info("WARNING: No val set!!! Computing metrics on test set!")
            eval_split = "test"
        logger.info("Evaluating on %s set:" % eval_split)
        test_metrics, test_preds = evaluate_split(
            model, loaders[eval_split], prefix="%s_" % eval_split, max_predictions=1000
        )
        str_out = "Results on train: "
        for k, v in train_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)
        str_out = "Results on %s: " % eval_split
        for k, v in test_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)

        # The main process is responsible for managing the checkpoint
        # TODO(gkioxari) revisit these stores
        """
        cp.store_metric(**train_preds)
        cp.store_metric(**test_preds)
        """
        cp.store_metric(**train_metrics)
        cp.store_metric(**test_metrics)
        cp.store_state("model", model.state_dict())
        cp.store_state("optim", optimizer.state_dict())
        cp.store_state("lr_scheduler", scheduler.state_dict())
        cp.save()

    # Since evaluation and checkpointing only happens on the main process,
    # make all processes wait
    if comm.get_world_size() > 1:
        dist.barrier()


def setup_loaders(cfg):
    loaders = {}
    loaders["train"] = build_data_loader(
        cfg, "MeshVox", "train", multigpu=comm.get_world_size() > 1
    )

    # Since sampling the mesh is now coupled with the data loader, we need to
    # make two different Dataset / DataLoaders for the training set: one for
    # training which uses precomputd samples, and one for evaluation which uses
    # more samples and computes them on the fly. This is sort of gross.
    loaders["train_eval"] = build_data_loader(cfg, "MeshVox", "train_eval", multigpu=False)

    loaders["val"] = build_data_loader(cfg, "MeshVox", "val", multigpu=False)
    return loaders


def setup(args):
    """
    Create configs and setup logger from arguments and the given config file.
    """
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # register dataset
    data_dir, splits_file = register_shapenet(cfg.DATASETS.NAME)
    cfg.DATASETS.DATA_DIR = data_dir
    cfg.DATASETS.SPLITS_FILE = splits_file
    # if data was copied the data dir has changed
    if args.copy_data:
        cfg.DATASETS.DATA_DIR = args.data_dir
    cfg.freeze()

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comm.synchronize()

    logger = setup_logger(
        output_dir, color=colorful_logging, name="shapenet", distributed_rank=comm.get_rank()
    )
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def shapenet_launch():
    args = default_argument_parser()

    # Note we need this only for pretrained models with torchvision.
    os.environ["TORCH_HOME"] = args.torch_home

    if args.copy_data:
        # if copy data is 1 then you need to provide args.data_dir
        # from which to copy data
        if args.data_dir == "":
            raise ValueError("You need to provide args.data_dir")
        copy_data(args)

    if args.eval_only:
        main_worker_eval(0, args)
        return

    if args.num_gpus > 1:
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,), daemon=False)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    shapenet_launch()
