"""Clipped PPO and factual-shift losses against specs/ppo.md (3)-(4)."""

import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.ppo.loss import ppo_loss, shift_loss

F64 = torch.float64


def probabilities(row):
    peak = max(row)
    denominator = sum(math.exp(value - peak) for value in row)
    return [math.exp(value - peak) / denominator for value in row]


def oracle_row(row, action, behavior, advantage, clip):
    log_probability = math.log(probabilities(row)[action])
    ratio = math.exp(log_probability) / behavior
    clipped = min(max(ratio, 1.0 - clip), 1.0 + clip)
    return -min(ratio * advantage, clipped * advantage)


def test_all_clipping_regions_for_positive_negative_and_zero_advantages():
    ratios = [0.5, 0.8, 1.0, 1.2, 1.5]
    advantages = [2.0, -2.0, 0.0]
    rows = [(advantage, ratio) for advantage in advantages for ratio in ratios]
    logits = torch.zeros(len(rows), 2, dtype=F64)
    actions = torch.tensor([i % 2 for i in range(len(rows))], dtype=torch.int64)
    behavior = torch.tensor([0.5 / ratio for _, ratio in rows], dtype=F64)
    advantage = torch.tensor([value for value, _ in rows], dtype=F64)
    got = ppo_loss(logits, actions, behavior, advantage, 0.2)
    want = [
        oracle_row([0.0, 0.0], int(actions[i]), 0.5 / ratio, value, 0.2)
        for i, (value, ratio) in enumerate(rows)
    ]
    assert got.shape == (len(rows),) and got.dtype == F64
    assert_close(got, torch.tensor(want, dtype=F64))


def test_full_unmasked_softmax_action_indexing_and_stable_ratio():
    logits = torch.tensor(
        [[1000.0, 0.0, -1000.0], [0.2, -0.4, 1.1], [-2.0, 0.5, 3.0]],
        dtype=F64,
    )
    actions = torch.tensor([0, 2, 1], dtype=torch.int64)
    behavior = torch.tensor([0.25, 0.4, 0.7], dtype=F64)
    advantages = torch.tensor([1.3, -0.8, 0.0], dtype=F64)
    got = ppo_loss(logits, actions, behavior, advantages, 0.2)
    want = [
        oracle_row(row.tolist(), int(action), float(mu), float(adv), 0.2)
        for row, action, mu, adv in zip(logits, actions, behavior, advantages)
    ]
    assert torch.isfinite(got).all()
    assert_close(got, torch.tensor(want, dtype=F64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ppo_loss_dtype(dtype):
    logits = torch.tensor([[0.3, -0.7, 1.2], [2.0, 0.0, -1.0]], dtype=dtype)
    actions = torch.tensor([2, 0], dtype=torch.int64)
    behavior = torch.tensor([0.3, 0.6], dtype=dtype)
    advantages = torch.tensor([1.0, -2.0], dtype=dtype)
    got = ppo_loss(logits, actions, behavior, advantages, 0.2)
    want = [
        oracle_row(row.tolist(), int(action), float(mu), float(adv), 0.2)
        for row, action, mu, adv in zip(logits, actions, behavior, advantages)
    ]
    tolerance = {"rtol": 1e-6, "atol": 1e-6} if dtype == torch.float32 else {}
    assert got.dtype == dtype
    assert_close(got, torch.tensor(want, dtype=dtype), **tolerance)


def test_logit_gradients_and_detached_behavior_and_advantages():
    base = torch.tensor(
        [[0.2, -0.1, 0.7], [1.0, 0.5, -0.5], [0.0, 2.0, 1.0]], dtype=F64
    )
    logits = base.clone().requires_grad_()
    actions = torch.tensor([2, 0, 1], dtype=torch.int64)
    probs = [probabilities(row.tolist()) for row in base]
    behavior = torch.tensor(
        [probs[i][int(actions[i])] for i in range(3)], dtype=F64, requires_grad=True
    )
    advantages = torch.tensor([1.5, -2.0, 0.0], dtype=F64, requires_grad=True)
    ppo_loss(logits, actions, behavior, advantages, 0.2).sum().backward()
    want = []
    for i, advantage in enumerate([1.5, -2.0, 0.0]):
        one_hot = [float(j == int(actions[i])) for j in range(3)]
        want.append([-advantage * (one_hot[j] - probs[i][j]) for j in range(3)])
    assert_close(logits.grad, torch.tensor(want, dtype=F64))
    assert behavior.grad is None and advantages.grad is None


@pytest.mark.parametrize("bound", [0.8, 1.2])
@pytest.mark.parametrize("advantage", [2.0, -2.0])
def test_exact_clipping_tie_splits_autograd_between_branches(bound, advantage):
    logits = torch.zeros(1, 2, dtype=F64, requires_grad=True)
    behavior = torch.tensor([0.5 / bound], dtype=F64)
    advantages = torch.tensor([advantage], dtype=F64)
    got = ppo_loss(
        logits, torch.tensor([0], dtype=torch.int64), behavior, advantages, 0.2
    )
    assert_close(got, torch.tensor([-bound * advantage], dtype=F64))
    got.backward()
    # At the bound clamp has zero derivative; an exact minimum tie gives each branch 1/2.
    want = [[-advantage * bound / 4.0, advantage * bound / 4.0]]
    assert_close(logits.grad, torch.tensor(want, dtype=F64))


def test_squared_metric_hand_values_for_every_factual_transition():
    predicted = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.0, 2.0]], dtype=F64)
    actual = torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.5, 0.5]], dtype=F64)
    alpha = torch.tensor([2.0, 0.5, 3.0], dtype=F64)
    got = shift_loss(
        predicted,
        actual,
        lambda delta: torch.linalg.vector_norm(alpha * delta, ord=2, dim=-1),
    )
    want = [
        sum(
            (float(alpha[j]) * (float(predicted[i, j]) - float(actual[i, j]))) ** 2
            for j in range(3)
        )
        for i in range(2)
    ]
    assert got.shape == (2,) and got.dtype == F64
    assert_close(got, torch.tensor(want, dtype=F64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shift_dtype_and_gradients_only_through_prediction(dtype):
    predicted = torch.tensor(
        [[1.0, -2.0, 0.5], [-1.0, 0.0, 2.0]], dtype=dtype, requires_grad=True
    )
    actual = torch.tensor(
        [[0.0, 1.0, -1.0], [0.5, 0.5, 0.5]], dtype=dtype, requires_grad=True
    )
    alpha = torch.tensor([2.0, 0.5, 3.0], dtype=dtype)
    got = shift_loss(
        predicted,
        actual,
        lambda delta: torch.linalg.vector_norm(alpha * delta, ord=2, dim=-1),
    )
    got.sum().backward()
    assert got.dtype == dtype and got.shape == (2,)
    assert_close(predicted.grad, 2.0 * alpha.square() * (predicted - actual))
    assert actual.grad is None
