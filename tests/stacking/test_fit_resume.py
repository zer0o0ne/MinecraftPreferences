"""`fit` of src/mcpref/stacking/fit.py interrupted (its `read` raising) and continued, against
specs/gui-stack.md (Behaviour (F); Produces: checkpoints, continuation; progress bars)."""

import math

import numpy as np
import pytest
import torch
from progress_bars import Bar, screen, states
from training.schedule_oracle import epoch

from . import fit_oracle as fo
from .fit_run import FIT, SIZES, checkpoint, checkpoints, fit_in

# M = 4, T = 12, N = 2: checkpoints inside an epoch (t = 2, 6, 10) and at its end (4, 8, 12)
CFG = FIT | {"epochs": 3, "checkpoint_steps": 2}
Q, M, T = len(SIZES), 4, 12
CALLS = Q + CFG["epochs"] * Q + Q  # reads: L(theta_0), the visits of (4), L(theta_T)


def last_step(k: int) -> int:
    """The step of the last checkpoint when the k-th read raises (0: none): the steps of the
    visits done before it, rounded down to a multiple of N."""
    rng, visits = np.random.Generator(np.random.PCG64(CFG["seed"])), []
    for _ in range(CFG["epochs"]):
        visits += dict.fromkeys(q for q, _ in epoch(rng, SIZES, CFG["batch"]))
    done = visits[: max(0, min(k - Q - 1, len(visits)))]
    steps = sum(math.ceil(SIZES[q] / CFG["batch"]) for q in done)
    return steps // CFG["checkpoint_steps"] * CFG["checkpoint_steps"]


def test_interruptions_cover_both_kinds():
    """The cases below leave the last checkpoint inside an epoch, at the end of one before T,
    at T, and none."""
    last = {last_step(k) for k in range(1, CALLS + 1)}
    assert 0 in last and T in last
    assert any(t % M for t in last) and any(t % M == 0 and 0 < t < T for t in last)


def expected_screen(last: int) -> list:
    """The bars of the continued fit: values in the restored checkpoint `already done`."""
    first = "loss initial: already done" if last else Bar("loss initial", Q, Q, "shard")
    fit = "stack fit: already done" if last == T else Bar("stack fit", T, T, "step")
    return [first, fit, Bar("loss final", Q, Q, "shard")]


@pytest.mark.parametrize("k", range(1, CALLS + 1))
def test_continued_fit_is_bitwise_equal(repo, tmp_path, k):
    """Interrupted at the k-th read, then continued: the same theta_T bit for bit, the same
    L(theta_0), L(theta_T) as an uninterrupted fit on the CPU; `stack fit` drawn from the
    step of the restored checkpoint."""
    z, y, w = fo.tiny_set(7)
    (want, first, last), _ = fit_in(
        repo, tmp_path / "whole", fo.Reader(z, y, w, SIZES), CFG
    )
    cut = tmp_path / "cut"
    with pytest.raises(fo.Interrupted):
        fit_in(repo, cut, fo.Reader(z, y, w, SIZES, fail=k), CFG)
    t = last_step(k)
    names = checkpoints(cut) if (cut / "checkpoints").exists() else []
    assert len(names) == t // CFG["checkpoint_steps"]
    if t:
        ck = checkpoint(cut, len(names) - 1)
        assert ck["step"] == t and ck["batch_in_epoch"] == t - (ck["epoch"] - 1) * M
    (got, first2, last2), err = fit_in(repo, cut, fo.Reader(z, y, w, SIZES), CFG)
    assert all(
        torch.equal(p.detach(), q.detach()) for p, q in zip(got, want, strict=True)
    )
    assert (first2, last2) == (first, last)
    assert screen(err) == expected_screen(t)
    if 0 < t < T:
        assert states(err, "stack fit")[0] == Bar("stack fit", t, T, "step")
