# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import torch
from detectron2.utils import comm as comm
from fvcore.common.file_io import PathManager
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from .mesh_vox import MeshVoxDataset

logger = logging.getLogger(__name__)


def _identity(x):
    return x


def build_data_loader(
    cfg, dataset, split_name, num_workers=4, multigpu=False, shuffle=True, num_samples=None
):

    batch_size = cfg.SOLVER.BATCH_SIZE
    return_mesh, sample_online, return_id_str = False, False, False
    if split_name in ["train_eval", "val"]:
        batch_size = cfg.SOLVER.BATCH_SIZE_EVAL
        return_mesh = True
        sample_online = True
    elif split_name == "test":
        batch_size = cfg.SOLVER.BATCH_SIZE_EVAL
        return_mesh = True
        return_id_str = True

    splits_file = cfg.DATASETS.SPLITS_FILE

    with PathManager.open(splits_file, "r") as f:
        splits = json.load(f)
    if split_name is not None:
        if split_name in ["train", "train_eval"]:
            split = splits["train"]
        else:
            split = splits[split_name]

    num_gpus = 1
    if multigpu:
        num_gpus = comm.get_world_size()
    assert batch_size % num_gpus == 0, "num_gpus must divide batch size"
    batch_size //= num_gpus

    logger.info('Building dataset for split "%s"' % split_name)
    if dataset == "MeshVox":
        dset = MeshVoxDataset(
            cfg.DATASETS.DATA_DIR,
            split=split,
            num_samples=cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
            voxel_size=cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE,
            return_mesh=return_mesh,
            sample_online=sample_online,
            return_id_str=return_id_str,
        )
        collate_fn = MeshVoxDataset.collate_fn
    else:
        raise ValueError("Dataset %s not registered" % dataset)

    loader_kwargs = {"batch_size": batch_size, "collate_fn": collate_fn, "num_workers": num_workers}

    if hasattr(dset, "postprocess"):
        postprocess_fn = dset.postprocess
    else:
        postprocess_fn = _identity

    # In this case we want to subsample num_samples elements from the underlying
    # dataset. We can wrap the dataset in a Subset dataset, so we need to tell
    # it which indices of the underlying dataset to use. Either take the first
    # or a random subset depending on whether shuffling was requested.
    if num_samples is not None:
        if shuffle:
            idx = torch.randperm(len(dset))[:num_samples]
        else:
            idx = torch.arange(min(num_samples, len(dset)))
        dset = Subset(dset, idx)

    # Right now we only do evaluation with a single GPU on the main process,
    # so only use a DistributedSampler for the training set.
    # TODO: Change this once we do evaluation on multiple GPUs
    if multigpu:
        assert shuffle, "Cannot sample sequentially with distributed training"
        sampler = DistributedSampler(dset)
    else:
        if shuffle:
            sampler = RandomSampler(dset)
        else:
            sampler = SequentialSampler(dset)
    loader_kwargs["sampler"] = sampler
    loader = DataLoader(dset, **loader_kwargs)

    # WARNING this is really gross! We want to access the underlying
    # dataset.postprocess method so we can run it on the main Python process,
    # but the dataset might be wrapped in a Subset instance, or may not even
    # define a postprocess method at all. To get around this we monkeypatch
    # the DataLoader object with the postprocess function we want; this will
    # be a bound method of the underlying Dataset, or an identity function.
    # Maybe it would be cleaner to subclass DataLoader for this?
    loader.postprocess = postprocess_fn

    return loader
