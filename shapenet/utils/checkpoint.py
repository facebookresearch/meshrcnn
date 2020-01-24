# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import torch

logger = logging.getLogger(__name__)


class Checkpoint(object):
    # These keys are saved in all checkpoints
    KEYS_TO_SAVE = [
        "t",
        "epoch",
        "metrics",
        "metrics_ts",
        "data",
        "early_stop_metric",
        "with_model_path",
        "no_model_path",
        "restarts",
    ]

    # These keys are saved for "big" checkpoints that include the model state
    STATE_KEYS = ["latest_states", "latest_states_ts", "best_states", "best_states_ts"]

    def __init__(self, output_path="checkpoint.pt", early_stop_metric=None, overwrite=False):
        output_dir, filename = os.path.split(output_path)
        filename, ext = os.path.splitext(filename)
        self.with_model_path = "%s_with_model%s" % (filename, ext)
        self.with_model_path = os.path.join(output_dir, self.with_model_path)
        self.no_model_path = "%s_no_model%s" % (filename, ext)
        self.no_model_path = os.path.join(output_dir, self.no_model_path)

        self.t = 0
        self.epoch = 0

        # Metrics change over time, data doesn't
        self.metrics = {}
        self.metrics_ts = {}
        self.data = {}

        self.latest_states = {}
        self.latest_states_ts = {}
        self.best_states = {}
        self.best_states_ts = {}
        self.early_stop_metric = early_stop_metric

        self.restarts = []
        if os.path.isfile(self.with_model_path) and not overwrite:
            logger.info('Loading checkpoint from "%s"' % self.with_model_path)
            self.from_dict(torch.load(self.with_model_path))
            self.restarts.append(self.t)

    def step(self):
        self.t += 1

    def step_epoch(self):
        self.epoch += 1

    def store_data(self, k, v):
        self.data[k] = v

    def store_metric(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.metrics:
                self.metrics[k] = []
                self.metrics_ts[k] = []
            self.metrics[k].append(v)
            self.metrics_ts[k].append(self.t)

    def store_state(self, name, state, best=None):
        self.latest_states[name] = state
        self.latest_states_ts[name] = self.t

        if best is None:
            k = self.early_stop_metric
            if k not in self.metrics:
                best = True
            else:
                max_v = max(self.metrics[k])
                last_v = self.metrics[k][-1]
                last_t = self.metrics_ts[k][-1]
                if self.t == last_t and last_v == max_v:
                    best = True
                else:
                    best = False

        if best is None:
            raise ValueError("Cannot determine whether current state is best")

        if best:
            logger.info('Storing new best state for "%s"' % name)
            self.best_states[name] = state
            self.best_states_ts[name] = state

    def to_dict(self, include_states=False):
        keys = [k for k in self.KEYS_TO_SAVE]
        if include_states:
            keys += self.STATE_KEYS
        d = {k: getattr(self, k) for k in keys}
        return d

    def from_dict(self, d):
        for k in d.keys():
            setattr(self, k, d[k])

    def save(self):
        logger.info('Saving checkpoint (with model) to "%s"' % self.with_model_path)
        torch.save(self.to_dict(include_states=True), self.with_model_path)

        logger.info('Saving checkpoint (without model) to "%s"' % self.no_model_path)
        torch.save(self.to_dict(include_states=False), self.no_model_path)


def clean_state_dict(state_dict):
    # Ugly hack to clean up the state dict in case we forgot to unpack the
    # underlying model from DistributedDataParallel when training
    out = {}
    for k, v in state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out
