"""Causal action alignment checked on distinct adjacent transitions."""

import math

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.models.policy.targets import next_actions, targets, weights
from models.policy.policy_oracle import cfg_of
from models.policy.policy_world import NAMES, make_clip


def test_each_state_consumes_previous_action_and_predicts_own_interval(masks):
    clip = make_clip([["attack"], ["use"], ["jump"]], [0.0] * 4, [0] * 4)
    index = {name: i for i, name in enumerate(masks.names)}
    current = weights(clip, index)
    expected = np.zeros_like(current)
    expected[0, -1] = 1
    for state, name in enumerate(("attack", "use", "jump"), start=1):
        expected[state, index[name]] = 1
    np.testing.assert_array_equal(current, expected)
    q, valid = targets(clip, masks)
    assert q.shape == (3, len(index))
    assert q.argmax(-1).tolist() == [index[name] for name in ("attack", "use", "jump")]
    assert valid.tolist() == [True, True, True]
    assert next_actions(clip, index).tolist() == [
        index["attack"],
        index["use"],
        index["jump"],
        0,
    ]


def test_current_action_cannot_leak_into_its_logits():
    clip = make_clip([["attack"], ["use"], ["jump"]], [0.0] * 4, [0] * 4)
    changed = make_clip([["attack"], ["forward"], ["jump"]], [0.0] * 4, [0] * 4)
    net = Policy(cfg_of(history=4), 3, NAMES).double().eval()
    latents = torch.randn(
        1, 4, 3, generator=torch.Generator().manual_seed(51), dtype=torch.float64
    )
    before, _, _, _ = net(latents, [clip])
    after, _, _, _ = net(latents, [changed])
    assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert not torch.allclose(before[:, 2], after[:, 2])


def test_known_idle_uses_no_action_on_its_successor(masks):
    clip = make_clip([[], ["attack"]], [0.0] * 3, [0] * 3)
    index = {name: i for i, name in enumerate(masks.names)}
    current = weights(clip, index)
    assert current.argmax(-1).tolist() == [len(index), len(index), index["attack"]]
    np.testing.assert_array_equal(current.sum(-1), np.ones(3))


def test_shift_validity_depends_only_on_own_single_allowed_action(masks):
    clip = make_clip(
        [["attack", "use"], ["jump"], ["attack", "use"]], [0.0] * 4, [0] * 4
    )
    _, valid = targets(clip, masks)
    assert valid.tolist() == [False, True, False]


def test_final_transition_has_loss_and_gradient_with_all_transition_denominator(masks):
    clip = make_clip([["attack"], ["use"]], [0.0] * 3, [0] * 3)
    logits = torch.zeros(
        1, 3, len(masks.names), dtype=torch.float64, requires_grad=True
    )
    shifts = torch.tensor([[[0.0], [3.0], [100.0]]], requires_grad=True)
    latents = torch.tensor([[[0.0], [0.0], [1.0]]], requires_grad=True)
    loss = PolicyLoss(
        cfg_of(shift_weight=1.5), masks, lambda delta: delta[..., 0].abs()
    )
    value = loss((logits, shifts), latents, [clip])
    assert value.item() == pytest.approx(math.log(len(masks.names)) + 3)
    value.sum().backward()
    assert shifts.grad.tolist() == [[[0.0], [3.0], [0.0]]]
    assert latents.grad is None
    assert logits.grad[0, 1, masks.names.index("use")] < 0
    assert not logits.grad[0, 2].any()
