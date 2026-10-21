"""Transcription of the mini-batch schedule of specs/gui-stack.md (Notation) and
specs/pretrain.md (3), nothing from src/: one epoch of a generator, and $M$."""

import math

import numpy as np


def epoch(rng: np.random.Generator, sizes: list, batch: int) -> list:
    """One epoch: pi_e = permutation(Q); for each shard q in that order, permutation(|S_q|)
    cut into consecutive mini-batches of B frames: (q, positions in S_q)."""
    out = []
    for q in rng.permutation(len(sizes)):
        perm = rng.permutation(sizes[q])
        out += [(int(q), perm[k : k + batch]) for k in range(0, sizes[q], batch)]
    return out


def mini_batches(sizes: list, batch: int) -> int:
    """M = sum_q ceil(|S_q| / B)."""
    return sum(math.ceil(n / batch) for n in sizes)
