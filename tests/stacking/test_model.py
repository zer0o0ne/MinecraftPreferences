"""src/mcpref/stacking/model.py against specs/gui-stack.md ((1), (2), (3'), (5)), with the
float64 references of stack_oracle.py."""

import math

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.stacking.model import init_params, logits, loss, probabilities

from . import stack_oracle as so

CPU = torch.device("cpu")


def t64(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def t32(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def tiny_case(seed: int) -> tuple:
    """C = 3, H = 4, |Y| = 4: theta and z of 7 frames (float32 values as float64), label
    indices y with label 1 absent, w_f = 1 / n_{y_f} (sum 3)."""
    rng = np.random.default_rng(seed)
    shapes = ((4, 3), (4,), (4, 4), (4,))
    theta = [rng.normal(size=s).astype(np.float32).astype(np.float64) for s in shapes]
    z = rng.normal(size=(7, 3)).astype(np.float32).astype(np.float64)
    y = [0, 2, 0, 3, 0, 2, 3]
    return theta, z, y, so.weights(y)


SIZES = [(3, 4, 5, 0), (1, 1, 2, 3), (6, 2, 3, 2**64 - 1)]  # C, H, |Y|, seed


@pytest.mark.parametrize("c, h, n, seed", SIZES)
def test_init_shapes(c, h, n, seed):
    """(2): [W1, b1, W2, b2], float32, on the device, requiring grad."""
    params = init_params(c, h, n, seed, CPU)
    assert [tuple(p.shape) for p in params] == [(h, c), (h,), (n, h), (n,)]
    assert all(p.dtype == torch.float32 and p.device == CPU for p in params)
    assert all(p.requires_grad for p in params)


@pytest.mark.parametrize("c, h, n, seed", SIZES)
def test_init_draws_of_one_generator(c, h, n, seed):
    """(2): (2u - 1) / sqrt(fan_in), u = torch.rand of each shape in the order W1, b1, W2, b2
    from one CPU generator seeded with `seed`; the global RNG plays no part."""
    torch.manual_seed(12345)
    got = init_params(c, h, n, seed, CPU)
    for p, want in zip(got, so.init(c, h, n, seed)):
        assert_close(p.detach(), torch.tensor(want, dtype=torch.float32))


def test_init_bounds():
    """(2): every value in [-1/sqrt(fan_in), 1/sqrt(fan_in)); fan_in C for W1, b1, H for W2,
    b2; the draws fill the range."""
    c, h = 16, 200
    params = init_params(c, h, 400, 1, CPU)
    for p, fan in zip(params, (c, c, h, h)):
        p = p.detach().double() * math.sqrt(fan)
        assert -1 - 1e-6 <= p.min() < -0.9 and 0.9 < p.max() < 1 + 1e-6


def test_init_same_seed_same_values():
    """(2): the same seed, the same values, whatever the global RNG did; another seed, others."""
    torch.manual_seed(0)
    a = init_params(3, 4, 5, 11, CPU)
    torch.rand(100)
    b = init_params(3, 4, 5, 11, CPU)
    c = init_params(3, 4, 5, 12, CPU)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not any(torch.equal(p, q) for p, q in zip(a, c))


HAND = [  # W1 (3 x 2), b1, W2 (2 x 3), b2
    [[1.0, -1.0], [0.5, 2.0], [-1.0, -1.0]],
    [0.0, -1.0, 0.5],
    [[1.0, 0.0, 2.0], [-1.0, 1.0, 0.0]],
    [0.25, -0.5],
]


def test_logits_hand_case():
    """(1): z = (1, 2): W1 z + b1 = (-1, 3.5, -2.5), h = (0, 3.5, 0), l = (0.25, 3);
    z = (3, -1): W1 z + b1 = (4, -1.5, -1.5), h = (4, 0, 0), l = (4.25, -4.5)."""
    got = logits([t64(p) for p in HAND], t64([[1.0, 2.0], [3.0, -1.0]]))
    assert_close(got, t64([[0.25, 3.0], [4.25, -4.5]]))


def test_logits_naive_reference():
    theta, z, _, _ = tiny_case(1)
    assert_close(logits([t64(p) for p in theta], t64(z)), t64(so.logits(theta, z)))


BATCH = [1, 3, 5]  # labels 2, 3, 2: label 0 of S absent from the mini-batch


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_loss_mini_batch_naive_reference(l2):
    """(3') in float32: M sum_B w_f (-log p(y_f | z_f)) / sum_S w_f + L2, softmax over all of
    Y, with M = 4 and sum_S w = 3 (not sum_B w, not |B|), within a relative 1e-5."""
    theta, z, y, w = tiny_case(2)
    yb = [y[f] for f in BATCH]
    got = loss([t32(p) for p in theta], t32(z[BATCH]), torch.tensor(yb), t32(w[BATCH]),
               3.0, 4, l2)  # fmt: skip
    want = so.loss_batch(theta, z[BATCH], yb, w[BATCH], 3.0, 4, l2)
    assert got.item() == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_loss_epoch_mean_is_full_loss(l2):
    """(3'): at a fixed theta, its mean over the M = 4 mini-batches of an epoch (shards of
    frames 0-3 and 4-6, B = 2; one-label mini-batches included) is (3) over S."""
    theta, z, y, w = tiny_case(3)
    batches = [[2, 0], [3, 1], [6, 4], [5]]
    losses = [
        loss([t32(p) for p in theta], t32(z[b]), torch.tensor([y[f] for f in b]),
             t32(w[b]), 3.0, 4, l2).item()
        for b in batches
    ]  # fmt: skip
    assert np.mean(losses) == pytest.approx(so.loss(theta, z, y, w, l2), rel=1e-5)


def test_loss_gradient_naive_reference():
    """(3'): the autograd gradient equals the hand gradient (coefficient M / sum_S w)."""
    theta, z, y, w = tiny_case(4)
    params = [t64(p).requires_grad_() for p in theta]
    yb = [y[f] for f in BATCH]
    value = loss(params, t64(z[BATCH]), torch.tensor(yb), t64(w[BATCH]), 3.0, 4, 0.2)
    got = torch.autograd.grad(value, params)
    for g, want in zip(got, so.gradient(theta, z[BATCH], yb, w[BATCH], 4 / 3.0, 0.2)):
        assert_close(g, t64(want))


def test_l2_term_on_weights_only():
    """(3'): L(l2) - L(0) = l2 (||W1||^2 + ||W2||^2); its gradient 2 l2 W1, 0, 2 l2 W2, 0."""
    theta, z, y, w = tiny_case(5)
    params = [t64(p).requires_grad_() for p in theta]

    def at(l2):
        value = loss(params, t64(z), torch.tensor(y), t64(w), 3.0, 1, l2)
        return value, torch.autograd.grad(value, params)

    (l0, g0), (l1, g1) = at(0.0), at(0.25)
    norms = sum((p**2).sum() for p in (theta[0], theta[2]))
    assert_close(l1 - l0, torch.tensor(0.25 * norms, dtype=torch.float64))
    for a, b, p, on in zip(g1, g0, theta, (1, 0, 1, 0)):
        assert_close(a - b, t64(2 * 0.25 * p * on))


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
