"""src/mcpref/stacking/model.py against specs/gui-stack.md ((4) `fit`, (5) `probabilities`),
with the float64 references of stack_oracle.py."""

import dataclasses
import math

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.stacking.config import load_config
from mcpref.stacking.model import fit, init_params, probabilities

from . import stack_oracle as so
from .stack_world import tiny_case

CPU = torch.device("cpu")


def config(repo, **changes):
    """The starting config with `changes`."""
    return dataclasses.replace(load_config(repo / "configs/gui-stack.yaml"), **changes)


def test_fit_adam_steps_naive_reference(repo):
    """(4): theta_1 .. theta_3 of full-batch Adam (bias-corrected moments, eps added to
    sqrt(v_hat)) against the float64 reference, in place; step() after each update; returns
    L(theta_0), L(theta_T)."""
    _, z, y, w = tiny_case(5)
    cfg = config(
        repo, hidden=4, l2=0.02, lr=0.05, betas=(0.8, 0.95), adam_eps=1e-3, steps=3
    )
    params = init_params(3, 4, 4, 9, CPU)
    theta0 = [p.detach().double().numpy() for p in params]
    snapshots = []
    z32 = torch.tensor(z, dtype=torch.float32)
    w32 = torch.tensor(w, dtype=torch.float32)

    def step():
        snapshots.append([p.detach().clone() for p in params])

    first, last = fit(params, z32, torch.tensor(y), w32, cfg, step)
    want = so.adam(theta0, z, y, w, 0.02, 0.05, (0.8, 0.95), 1e-3, 3)
    assert len(snapshots) == 3
    for got_t, want_t in zip(snapshots, want):
        for p, q in zip(got_t, want_t):
            assert_close(p, torch.tensor(q, dtype=torch.float32))
    assert all(torch.equal(p.detach(), q) for p, q in zip(params, snapshots[-1]))
    assert first == pytest.approx(so.loss(theta0, z, y, w, 0.02), rel=1e-5)
    assert last == pytest.approx(so.loss(want[-1], z, y, w, 0.02), rel=1e-5)


def test_fit_separates_a_separable_set(repo):
    """(4) with the starting config but T = 300: no training error on three separated
    clusters of 9, 3 and 1 frames; the loss decreases."""
    rng = np.random.default_rng(6)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    y = [0] * 9 + [1] * 3 + [2]
    x = centres[y] + rng.normal(scale=0.3, size=(13, 2))
    mu, s = so.moments(x)
    z = torch.tensor(so.standardised(x, mu, s), dtype=torch.float32)
    w = torch.tensor(so.weights(y), dtype=torch.float32)
    cfg = config(repo, steps=300)
    params = init_params(2, cfg.hidden, 3, cfg.seed, CPU)
    first, last = fit(params, z, torch.tensor(y), w, cfg, lambda: None)
    assert last < first
    assert np.argmax(probabilities(params, z), axis=1).tolist() == y


def test_probabilities_naive_reference():
    """(5): softmax of l over Y, frames x |Y|, float64, rows summing to 1."""
    params = init_params(3, 4, 5, 2, CPU)
    z = np.random.default_rng(7).normal(size=(6, 3))
    got = probabilities(params, torch.tensor(z, dtype=torch.float32))
    theta = [p.detach().double().numpy() for p in params]
    want = np.exp(so.log_softmax(so.logits(theta, z)))
    assert isinstance(got, np.ndarray) and got.dtype == np.float64
    assert got.shape == (6, 5)
    assert_close(torch.tensor(got), torch.tensor(want), rtol=1e-5, atol=1e-8)
    assert_close(torch.tensor(got.sum(axis=1)), torch.ones(6, dtype=torch.float64))


def test_probabilities_softmax_in_float64():
    """(5): l = (0, -120) exactly in float32; p_2 = e^-120 / (1 + e^-120) ~ 7.7e-53, which a
    float32 softmax flushes to 0."""
    params = [torch.tensor(v, dtype=torch.float32) for v in
              ([[1.0]], [0.0], [[0.0], [0.0]], [0.0, -120.0])]  # fmt: skip
    got = probabilities(params, torch.tensor([[0.5]], dtype=torch.float32))
    e = math.exp(-120)
    assert got[0, 1] == pytest.approx(e / (1 + e), rel=1e-12)
    assert got[0, 0] == pytest.approx(1 / (1 + e), rel=1e-15)
