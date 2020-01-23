# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        total_iters,
        warmup_iters=500,
        warmup_factor=0.1,
        eta_min=0.0,
        last_epoch=-1,
        warmup_method="cosine",
    ):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        assert warmup_method in ["linear", "cosine"]
        self.warmup_method = warmup_method
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                lr_factor = self.warmup_factor * (1 - alpha) + alpha
            elif self.warmup_method == "cosine":
                t = 1.0 + self.last_epoch / self.warmup_iters
                cos_factor = (1.0 + math.cos(math.pi * t)) / 2.0
                lr_factor = self.warmup_factor + (1.0 - self.warmup_factor) * cos_factor
            else:
                raise ValueError("Unsupported warmup method")
            return [lr_factor * base_lr for base_lr in self.base_lrs]

        num_decay_iters = self.total_iters - self.warmup_iters
        t = (self.last_epoch - self.warmup_iters) / num_decay_iters
        cos_factor = (1.0 + math.cos(math.pi * t)) / 2.0
        lrs = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * cos_factor
            lrs.append(lr)
        return lrs


class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]
