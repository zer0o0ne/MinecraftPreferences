"""src/mcpref/stacking/fit.py against specs/gui-stack.md ((3), (4); Behaviour (F);
Produces: checkpoints; progress bars), with the float64 transcriptions of fit_oracle.py; the
tests of `schedule` are in tests/training/test_schedule.py."""

import io

import numpy as np
import pytest
import torch
from progress_bars import Bar, assert_progress, screen
from torch.testing import assert_close
from training.schedule_oracle import epoch

from mcpref.stacking.fit import fit, full_loss
from mcpref.stacking.model import init_params, probabilities

from . import fit_oracle as fo
from . import stack_oracle as so
from .fit_run import FIT, SIZES, checkpoint, checkpoints, config, fit_in, oracle

CPU = torch.device("cpu")


def test_full_loss_naive_reference():
    """(3) over three shards against the oracle on their concatenation (rel 1e-5): each
    shard read once; bar `desc` (unit shard, total q)."""
    z, y, w = fo.tiny_set(1)
    params = init_params(3, 4, 4, 9, CPU)
    read = fo.Reader(z, y, w, [3, 2, 2])
    err = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.stderr", err)
        got = full_loss(params, read, 3, 0.02, "loss check")
    theta = [p.detach().double().numpy() for p in params]
    assert type(got) is float and got == pytest.approx(
        so.loss(theta, z, y, w, 0.02), rel=1e-5
    )
    assert sorted(read.calls) == [0, 1, 2]
    assert_progress(err.getvalue())
    assert screen(err.getvalue()) == [Bar("loss check", 3, 3, "shard")]


def test_fit_every_step_naive_reference(repo, tmp_path):
    """(4) with a checkpoint after every step (N = 1): theta_t, Adam's m_t, v_t and t of
    checkpoint t - 1 against the float64 transcription; theta_T returned; L(theta_0),
    L(theta_T) of (3) (rel 1e-5)."""
    z, y, w = fo.tiny_set(2)
    (params, first, last), _ = fit_in(repo, tmp_path, fo.Reader(z, y, w, SIZES), FIT)
    theta0, path = oracle(z, y, w, FIT)
    assert checkpoints(tmp_path) == [f"{j:06d}" for j in range(8)]
    for j, (theta, m, v) in enumerate(path):
        ck = checkpoint(tmp_path, j)
        assert ck["step"] == j + 1
        for i, p in enumerate(ck["params"]):
            assert_close(p, torch.tensor(theta[i], dtype=torch.float32))
            state = ck["adam"]["state"][i]
            assert float(state["step"]) == j + 1
            assert_close(state["exp_avg"], torch.tensor(m[i], dtype=torch.float32))
            assert_close(state["exp_avg_sq"], torch.tensor(v[i], dtype=torch.float32))
    assert all(torch.equal(p.detach(), q) for p, q in zip(params, ck["params"]))
    assert first == pytest.approx(so.loss(theta0, z, y, w, FIT["l2"]), rel=1e-5)
    assert last == pytest.approx(so.loss(path[-1][0], z, y, w, FIT["l2"]), rel=1e-5)


def test_checkpoint_contents(repo, tmp_path):
    """N = 3, T = 8: checkpoints at t = 3 (epoch 1, 3 mini-batches done) and t = 6 (epoch 2,
    2 done) only; their keys and types; the schedule generator's state before the draws of
    the checkpoint's epoch; Adam's hyperparameters as the config and the Library mapping."""
    z, y, w = fo.tiny_set(3)
    cfg = FIT | {"checkpoint_steps": 3}
    (_, first, _), _ = fit_in(repo, tmp_path, fo.Reader(z, y, w, SIZES), cfg)
    _, states = fo.schedule(cfg["seed"], SIZES, cfg["batch"], cfg["epochs"])
    _, path = oracle(z, y, w, cfg)
    assert checkpoints(tmp_path) == ["000000", "000001"]
    keys = {"step", "params", "adam", "loss_initial", "epoch", "schedule_rng",
            "batch_in_epoch"}  # fmt: skip
    for j, (t, e, done) in enumerate([(3, 1, 3), (6, 2, 2)]):
        ck = checkpoint(tmp_path, j)
        assert set(ck) == keys
        assert (ck["step"], ck["epoch"], ck["batch_in_epoch"]) == (t, e, done)
        assert ck["schedule_rng"] == states[e - 1] and ck["loss_initial"] == first
        for p, want in zip(ck["params"], path[t - 1][0], strict=True):
            assert p.dtype == torch.float32 and p.device == CPU and not p.requires_grad
            assert_close(p, torch.tensor(want, dtype=torch.float32))
        group = ck["adam"]["param_groups"][0]
        assert (group["lr"], tuple(group["betas"]), group["eps"]) == (
            0.05,
            (0.8, 0.95),
            1e-3,
        )
        assert group["weight_decay"] == 0
        assert not (
            group["amsgrad"] or group["maximize"] or group["decoupled_weight_decay"]
        )


def test_fit_reads_each_shard_once_per_visit(repo, tmp_path):
    """read: each shard once for L(theta_0); then once per visit, in the order pi_e of every
    epoch; then each once for L(theta_T)."""
    z, y, w = fo.tiny_set(4)
    read = fo.Reader(z, y, w, SIZES)
    fit_in(repo, tmp_path, read, FIT)
    rng, visits = np.random.Generator(np.random.PCG64(FIT["seed"])), []
    for _ in range(FIT["epochs"]):
        visits += dict.fromkeys(q for q, _ in epoch(rng, SIZES, FIT["batch"]))
    assert sorted(read.calls[:2]) == sorted(read.calls[-2:]) == [0, 1]
    assert read.calls[2:-2] == visits


def test_fit_bars(repo, tmp_path):
    """Bars `loss initial` (Q shards), `stack fit` (T = EM steps), `loss final`, in order."""
    z, y, w = fo.tiny_set(5)
    _, err = fit_in(repo, tmp_path, fo.Reader(z, y, w, SIZES), FIT)
    assert_progress(err)
    assert screen(err) == [Bar("loss initial", 2, 2, "shard"),
                           Bar("stack fit", 8, 8, "step"), Bar("loss final", 2, 2, "shard")]  # fmt: skip


def test_fit_separates_a_separable_set(repo, tmp_path):
    """(4) with the starting config but E = 150 (shards of 7 and 6 frames, B = 32: T = 300):
    no training error on three separated clusters of 9, 3 and 1 frames; the loss decreases."""
    y = [0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0]
    noise = np.random.default_rng(6).normal(scale=0.3, size=(13, 2))
    x = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])[y] + noise
    z = so.standardised(x, *so.moments(x))
    read = fo.Reader(z, y, so.weights(y), [7, 6])
    cfg = config(repo, epochs=150)
    params, first, last = fit(cfg, read, [7, 6], 2, 3, 3.0, tmp_path / "checkpoints")
    assert last < first
    got = probabilities(params, torch.tensor(z, dtype=torch.float32))
    assert np.argmax(got, axis=1).tolist() == y
