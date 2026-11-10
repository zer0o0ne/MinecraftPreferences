"""Preference reward against specs/ppo.md (1)-(2), with independent float64 oracles."""

import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.ppo.reward import reward

F64 = torch.float64


class PairPreference(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.calls = []
        self.grad_modes = []
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=F64))

    def forward(self, consequence, origin):
        self.calls.append((consequence, origin))
        self.grad_modes.append(torch.is_grad_enabled())
        keys = [
            (float(consequence_row[-1, 0]), float(origin_row[-1, 0]))
            for consequence_row, origin_row in zip(consequence, origin)
        ]
        s, u = zip(*(self.logits[key] for key in keys))
        return (
            self.scale * torch.tensor(s, dtype=F64, device=consequence.device),
            self.scale * torch.tensor(u, dtype=F64, device=consequence.device),
        )


def history(identifier, length, requires_grad=False):
    value = torch.full((1, length, 2), float(identifier), dtype=F64)
    return value.requires_grad_(requires_grad)


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def oracle_row(origin, futures, logits, discount, min_p):
    total = 0.0
    for k, future in enumerate(futures, start=1):
        s, u = logits[(float(future[0, -1, 0]), float(origin[0, -1, 0]))]
        p_succ, p_inc = sigmoid(s), sigmoid(u)
        total += discount**k * (p_inc + min_p) * (2.0 * p_succ - 1.0)
    return total


def example_logits():
    return {
        (1.0, 10.0): (math.log(3.0), -0.5),
        (2.0, 10.0): (-1.25, 2.0),
        (3.0, 10.0): (0.75, 0.25),
        (4.0, 20.0): (2.5, -2.0),
        (5.0, 20.0): (-0.5, math.log(2.0)),
        (6.0, 20.0): (0.0, 1.5),
    }


def call_keys(call):
    consequence, origin = call
    return [
        (float(consequence_row[-1, 0]), float(origin_row[-1, 0]))
        for consequence_row, origin_row in zip(consequence, origin)
    ]


def test_every_consequence_first_pair_and_ascending_discounted_sum():
    logits = example_logits()
    origins = [history(10, 3), history(20, 3)]
    futures = [
        [history(1, 3), history(2, 3), history(3, 3)],
        [history(4, 3), history(5, 3), history(6, 3)],
    ]
    preference = PairPreference(logits)
    got = reward(preference, origins, futures, 3, 0.7, 0.4)
    want = [oracle_row(o, f, logits, 0.7, 0.4) for o, f in zip(origins, futures)]
    assert got.shape == (2,) and got.dtype == F64 and not got.requires_grad
    assert_close(got, torch.tensor(want, dtype=F64))
    assert len(preference.calls) == 1
    consequence, origin = preference.calls[0]
    assert consequence.shape == (6, 3, 2) and origin.shape == (6, 3, 2)
    assert call_keys(preference.calls[0]) == [
        (1.0, 10.0),
        (2.0, 10.0),
        (3.0, 10.0),
        (4.0, 20.0),
        (5.0, 20.0),
        (6.0, 20.0),
    ]


def test_unequal_lengths_are_grouped_and_restored_to_item_horizon_layout():
    logits = example_logits()
    origins = [history(10, 1), history(20, 4)]
    futures = [
        [history(1, 2), history(2, 5), history(3, 2)],
        [history(4, 3), history(5, 2), history(6, 3)],
    ]
    preference = PairPreference(logits)
    got = reward(preference, origins, futures, 3, 0.7, 0.4)
    want = [oracle_row(o, f, logits, 0.7, 0.4) for o, f in zip(origins, futures)]
    assert_close(got, torch.tensor(want, dtype=F64))
    assert [call_keys(call) for call in preference.calls] == [
        [(1.0, 10.0), (3.0, 10.0)],
        [(2.0, 10.0)],
        [(4.0, 20.0), (6.0, 20.0)],
        [(5.0, 20.0)],
    ]
    assert [call[0].shape[0] for call in preference.calls] == [2, 1, 2, 1]


def test_immediate_consequence_hand_value():
    # Both sigmoids are 3/4, so V=(3/4+1/4)(2*3/4-1)=1/2 and gamma*V=1/4.
    logits = {(2.0, 1.0): (math.log(3.0), math.log(3.0))}
    got = reward(
        PairPreference(logits),
        [history(1, 1)],
        [[history(2, 1)]],
        1,
        0.5,
        0.25,
    )
    assert got.item() == pytest.approx(0.25)


def test_complete_horizon_is_summed_without_normalisation_or_clipping():
    logits = {(float(k), 0.0): (math.log(3.0), math.log(3.0)) for k in (1, 2, 3)}
    got = reward(
        PairPreference(logits),
        [history(0, 1)],
        [[history(k, 1) for k in (1, 2, 3)]],
        3,
        1.0,
        0.25,
    )
    assert got.item() == pytest.approx(1.5)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("outer_grad", [False, True])
def test_no_grad_and_caller_modes_are_preserved(training, outer_grad):
    logits = {(2.0, 1.0): (0.3, -0.2), (3.0, 1.0): (-0.4, 0.8)}
    preference = PairPreference(logits).train(training)
    origin = history(1, 2, requires_grad=True)
    futures = [history(2, 1, requires_grad=True), history(3, 3, requires_grad=True)]
    rng_state = torch.random.get_rng_state()
    with torch.set_grad_enabled(outer_grad):
        got = reward(preference, [origin], [futures], 2, 0.9, 0.5)
        assert torch.is_grad_enabled() is outer_grad
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert preference.training is training
    assert preference.grad_modes == [False, False]
    assert got.grad_fn is None and not got.requires_grad
    assert preference.scale.requires_grad and preference.scale.grad is None
    assert origin.requires_grad and origin.grad is None
    assert all(future.requires_grad and future.grad is None for future in futures)
