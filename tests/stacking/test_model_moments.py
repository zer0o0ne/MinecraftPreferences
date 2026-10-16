"""`moments` and `standardised` of src/mcpref/stacking/model.py against specs/gui-stack.md
(Notation: mu, s, z; Behaviour (M): two passes over the shards of S), with the float64
oracle of stack_oracle.py on the concatenation of the shards."""

import warnings

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.stacking.model import moments, standardised

from . import stack_oracle as so

NAN = float("nan")
# Columns: NaN here and there; all NaN in shard 1; 0.1 everywhere (s = 0 exactly, though
# numpy.std of it is not 0); no value in S (mu = s = 0); 0.1 in shard 0 and 0.2 in shard 2
# (each shard constant, S not: s > 0); a single value (s = 0)
SHARDS = [
    np.array([[1.0, 4.0, 0.1, NAN, 0.1, NAN], [NAN, 2.0, 0.1, NAN, 0.1, 7.5]]),
    np.array([[3.0, NAN, 0.1, NAN, NAN, NAN]]),
    np.array([[2.5, 5.0, 0.1, NAN, 0.2, NAN], [NAN, NAN, 0.1, NAN, 0.2, NAN],
              [-1.0, 1.5, 0.1, NAN, NAN, NAN]]),
]  # fmt: skip


class Passes:
    """`passes` of `moments`: a fresh iterator over the shards' x on each call, counted."""

    def __init__(self, shards: list):
        self.shards, self.calls = shards, 0

    def __call__(self):
        self.calls += 1
        return iter([s.copy() for s in self.shards])


def t64(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def checked(shards: list) -> tuple:
    """moments over `shards` and the oracle on their concatenation, compared within a
    relative and an absolute 1e-12; s_i = 0 cases exactly."""
    passes = Passes(shards)
    mu, s = moments(passes)
    want_mu, want_s = so.moments(np.concatenate(shards))
    assert passes.calls == 2
    assert_close(t64(mu), t64(want_mu), rtol=1e-12, atol=1e-12)
    assert_close(t64(s), t64(want_s), rtol=1e-12, atol=1e-12)
    assert (s[want_s == 0] == 0).all()
    return mu, s


def test_moments_hand_shards(capfd):
    """Population std (divisor = count) over all shards, NaN ignored; no warning (an all-NaN
    column of a shard, a column without a value) and nothing printed."""
    assert np.std(np.full(6, 0.1)) != 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu, s = checked(SHARDS)
    assert capfd.readouterr().err == ""
    assert_close(t64(mu[[0, 1, 4, 5]]), t64([1.375, 3.125, 0.15, 7.5]))
    assert_close(t64(s[[0, 4]]), t64([np.sqrt(2.421875), 0.05]))
    assert s[[2, 3, 5]].tolist() == [0.0, 0.0, 0.0] and mu[3] == 0.0


def test_moments_column_without_a_value():
    """mu_i = 0, s_i = 0 when column i has no value in S (clarification of 2026-10-16)."""
    mu, s = moments(Passes([np.full((2, 2), NAN), np.array([[NAN, 1.0]])]))
    assert mu.tolist()[0] == 0.0 and s.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("seed", range(3))
def test_moments_naive_reference(seed):
    """Random shards of 1 to 6 rows, about 30 % NaN, a column constant at 0.3."""
    rng = np.random.default_rng(seed)
    shards = []
    for n in rng.integers(1, 7, size=4):
        x = rng.normal(size=(n, 5)) * 3 + 1
        x[rng.random(x.shape) < 0.3] = NAN
        x[:, 4] = 0.3
        shards.append(x)
    _, s = checked(shards)
    assert s[4] == 0.0


def test_standardised_hand_case():
    """z = (x - mu) / s; 0 where x is NaN (the mean) or s = 0, whatever x is there."""
    mu, s = np.array([2.0, 3.0, 0.1, 5.0, 0.0]), np.array([0.5, 1.0, 0.0, 0.0, 0.0])
    x = np.array([[4.0, NAN, 0.7, 9.0, 2.0], [NAN, 1.0, NAN, 5.0, NAN],
                  [1.0, 3.0, 0.1, -2.0, -1.0]])  # fmt: skip
    want = [
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -2.0, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert np.asarray(standardised(x, mu, s), dtype=np.float64).tolist() == want
