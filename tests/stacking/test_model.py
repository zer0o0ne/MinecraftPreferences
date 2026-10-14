"""src/mcpref/stacking/model.py against specs/gui-stack.md (Notation: mu, s, z; (1)-(3)),
with the float64 references of stack_oracle.py."""

import math

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.stacking.model import (
    init_params,
    logits,
    loss,
    moments,
    standardised,
)

from . import stack_oracle as so
from .stack_world import tiny_case

CPU = torch.device("cpu")
NAN = float("nan")
X = np.array(  # S x C: NaN; [1, 3] (population std 1); 0.1 three times; a single value
    [[1.0, NAN, 0.1, 5.0], [3.0, 2.0, 0.1, NAN], [NAN, 4.0, 0.1, NAN]]
)


def t64(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def test_moments_hand_case():
    """mu over the non-NaN values; s population (divisor = count); a constant column s = 0
    exactly, though numpy.nanstd of 0.1, 0.1, 0.1 is not 0."""
    assert np.nanstd(X[:, 2]) != 0
    mu, s = moments(X)
    assert_close(t64(mu), t64([2.0, 3.0, 0.1, 5.0]))
    assert s.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_moments_naive_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(9, 5)) * 3 + 1
    x[rng.random(x.shape) < 0.3] = NAN
    x[:, 4] = 0.7
    got, want = moments(x), so.moments(x)
    assert_close(t64(got[0]), t64(want[0]))
    assert_close(t64(got[1]), t64(want[1]))
    assert got[1][4] == 0.0


def test_standardised_hand_case():
    """z = (x - mu) / s; 0 where x is NaN (the mean) or s = 0, whatever x is there."""
    mu, s = np.array([2.0, 3.0, 0.1, 5.0]), np.array([0.5, 1.0, 0.0, 0.0])
    x = np.array([[4.0, NAN, 0.7, 9.0], [NAN, 1.0, NAN, 5.0], [1.0, 3.0, 0.1, -2.0]])
    want = [[4.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]
    assert np.asarray(standardised(x, mu, s), dtype=np.float64).tolist() == want


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


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_loss_naive_reference(l2):
    """(3) with a label absent from S, sum w = 3 (not 1, not |S|): the weighted mean over
    softmax over all of Y, plus l2 (||W1||^2 + ||W2||^2)."""
    theta, z, y, w = tiny_case(2)
    got = loss([t64(p) for p in theta], t64(z), torch.tensor(y), t64(w), l2)
    assert_close(got, torch.tensor(so.loss(theta, z, y, w, l2), dtype=torch.float64))


def test_loss_gradient_naive_reference():
    """(3): the autograd gradient equals the hand gradient of stack_oracle.py."""
    theta, z, y, w = tiny_case(3)
    params = [t64(p).requires_grad_() for p in theta]
    got = torch.autograd.grad(
        loss(params, t64(z), torch.tensor(y), t64(w), 0.2), params
    )
    for g, want in zip(got, so.gradient(theta, z, y, w, 0.2)):
        assert_close(g, t64(want))


def test_l2_term_on_weights_only():
    """(3): L(l2) - L(0) = l2 (||W1||^2 + ||W2||^2); its gradient 2 l2 W1, 0, 2 l2 W2, 0."""
    theta, z, y, w = tiny_case(4)
    params = [t64(p).requires_grad_() for p in theta]

    def at(l2):
        value = loss(params, t64(z), torch.tensor(y), t64(w), l2)
        return value, torch.autograd.grad(value, params)

    (l0, g0), (l1, g1) = at(0.0), at(0.25)
    norms = sum((p**2).sum() for p in (theta[0], theta[2]))
    assert_close(l1 - l0, torch.tensor(0.25 * norms, dtype=torch.float64))
    for a, b, p, on in zip(g1, g0, theta, (1, 0, 1, 0)):
        assert_close(a - b, t64(2 * 0.25 * p * on))
