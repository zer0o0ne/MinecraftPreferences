"""Transcriptions of (4) of specs/gui-stack.md ((3'), (4); its schedule from
tests/training/schedule_oracle.py) in float64, nothing from src/; the `read` of `fit` on
tiny shards."""

import math

import numpy as np
import torch
from training.schedule_oracle import epoch, mini_batches

from . import stack_oracle as so


def schedule(seed: int, sizes: list, batch: int, epochs: int) -> tuple[list, list]:
    """(4): the mini-batches of all epochs from numpy.random.Generator(PCG64(seed)), and the
    generator's bit_generator.state before the draws of each epoch."""
    rng = np.random.Generator(np.random.PCG64(seed))
    batches, states = [], []
    for _ in range(epochs):
        states.append(rng.bit_generator.state)
        batches += epoch(rng, sizes, batch)
    return batches, states


def adam_path(theta0: list, z, y: list, w, sizes: list, cfg: dict) -> list:
    """(4): (theta_t, m_t, v_t) for t = 1..T, Adam on the gradient of (3') of the t-th
    mini-batch of the schedule; z, y, w of S with its shards concatenated in order."""
    batches, _ = schedule(cfg["seed"], sizes, cfg["batch"], cfg["epochs"])
    offsets = np.cumsum([0, *sizes])
    coef = mini_batches(sizes, cfg["batch"]) / w.sum()
    (b1, b2), eps, lr = cfg["betas"], cfg["adam_eps"], cfg["lr"]
    theta, out = theta0, []
    m = [np.zeros_like(p) for p in theta]
    v = [np.zeros_like(p) for p in theta]
    for t, (q, pos) in enumerate(batches, 1):
        idx = offsets[q] + pos
        g = so.gradient(theta, z[idx], [y[i] for i in idx], w[idx], coef, cfg["l2"])
        m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(m, g)]
        v = [b2 * vi + (1 - b2) * gi**2 for vi, gi in zip(v, g)]
        theta = [
            p - lr * (mi / (1 - b1**t)) / (np.sqrt(vi) / math.sqrt(1 - b2**t) + eps)
            for p, mi, vi in zip(theta, m, v)
        ]
        out.append((theta, m, v))
    return out


def tiny_set(seed: int) -> tuple:
    """S of 7 frames in shards of 5 and 2, C = 3, |Y| = 4, label 1 absent: z (float32 values
    as float64), label indices y, w_f = 1 / n_{y_f} (sum 3)."""
    z = np.random.default_rng(seed).normal(size=(7, 3)).astype(np.float32)
    y = [0, 2, 0, 3, 0, 2, 3]
    return z.astype(np.float64), y, so.weights(y)


class Interrupted(Exception):
    """Raised by Reader at its `fail`-th call."""


class Reader:
    """`read(r)` of `fit`: z, y, w of shard r (float32, int64, float32 on the CPU); records
    each r in `calls`; raises Interrupted at call number `fail` (1-based)."""

    def __init__(self, z, y: list, w, sizes: list, fail: int | None = None):
        self.z, self.y, self.w, self.fail = z, y, w, fail
        self.offsets, self.calls = np.cumsum([0, *sizes]), []

    def __call__(self, r: int) -> tuple:
        self.calls.append(r)
        if len(self.calls) == self.fail:
            raise Interrupted
        lo, hi = self.offsets[r], self.offsets[r + 1]
        return (
            torch.tensor(self.z[lo:hi], dtype=torch.float32),
            torch.tensor(self.y[lo:hi], dtype=torch.int64),
            torch.tensor(self.w[lo:hi], dtype=torch.float32),
        )
