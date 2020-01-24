# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
""" Utilities for timing GPU operations in PyTorch. """
import logging
import numpy as np
import time
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)


def time_backward(f, x, key=None, timing=None):
    """
    Utility function for timing the backward pass. Suppose we have the operation
    y = f(x) and we want to know how long the backward pass will take. We can
    then write:

    y = time_backward(f, x, 'f')

    This will set up backward hooks in the graph that start a Timer once grad_y
    has been computed, and stop the Timer when grad_x has been computed.
    """
    if callable(f):
        y = f(x)
    else:
        y = f
    timer = Timer(key=key, timing=timing)

    def y_hook(_grad_y):
        timer.start()

    def x_hook(_grad_x):
        timer.stop()

    if y.requires_grad and x.requires_grad:
        y.register_hook(y_hook)
        x.register_hook(x_hook)
    return y


def timeit(f, x, key=None, timing=None):
    """
    Utility function that times both the forward and backward pass of y = f(x).
    """
    f_key = "%s-forward" % key
    b_key = "%s-backward" % key
    with Timer(f_key, timing):
        y = time_backward(f, x, b_key, timing)
    return y


class Timer(object):
    """
    A context manager for timing nested chunks of code, like this:

    with Timer('my_loop'):
        out = 0
        for x in range(100):
            with Timer('my_op'):
                out += f(x)

    If you set Timer.timing = True then this will print mean and std dev timing
    for both my_loop and my_op.
    """

    _indent_level = 0
    timing = False
    _times = defaultdict(list)

    @classmethod
    def _adjust_indent(cls, val):
        cls._indent_level += val

    @classmethod
    def _record_time(cls, key, val):
        cls._times[key].append(val)

    @classmethod
    def get_stats(cls, key):
        times = cls._times[key]
        return np.mean(times), np.std(times)

    @classmethod
    def reset(cls):
        cls._times = defaultdict(list)

    def __init__(self, key=None, timing=None):
        self._key = key
        self._local_timing = timing

    def _should_time(self):
        if self._local_timing is not None:
            return self._local_timing
        return self.timing

    def start(self):
        if self._should_time():
            self._adjust_indent(1)
            torch.cuda.synchronize()
            self._t0 = time.time()

    def stop(self):
        if self._should_time():
            torch.cuda.synchronize()
            self._t1 = time.time()
            duration_ms = (self._t1 - self._t0) * 1000.0
            key = self._key
            space = "  " * self._indent_level
            if key is not None:
                self._record_time(key, duration_ms)
                mean, std = self.get_stats(key)
                msg = "[timeit]%s%s: %.4f ms (mean=%.4f ms, std=%.4f ms)" % (
                    space,
                    key,
                    duration_ms,
                    mean,
                    std,
                )
            else:
                msg = "[timeit]%s%.4f" % (space, duration_ms)
            logger.info(msg)
            self._adjust_indent(-1)

    def tick(self):
        self.stop()
        self.start()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.stop()
