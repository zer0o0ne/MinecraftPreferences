"""Independent values and gradient directions of the action-change mechanism."""

import math

import pytest
import torch

from mcpref.models.policy.change import change_cross_entropy, change_loss
from mcpref.models.policy.network import Policy
from models.policy.policy_oracle import cfg_of


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("loss", [-3.0, 0.0, 2.0])
def test_change_loss_preserves_action_direction_and_learns_the_change(repeated, loss):
    probability = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    action_loss = torch.tensor(loss, dtype=torch.float64, requires_grad=True)
    result = change_loss(probability, torch.tensor(repeated), action_loss)
    expected_error = 0.8 if repeated else 0.2
    assert result.item() == pytest.approx(expected_error * abs(loss))
    result.backward()
    assert action_loss.grad.item() == pytest.approx(expected_error)
    assert probability.grad.item() == pytest.approx(abs(loss) * (1 if repeated else -1))


def test_per_action_weighting_precedes_reduction():
    probability = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    losses = torch.tensor([1.0, 4.0], dtype=torch.float64, requires_grad=True)
    result = change_loss(probability, torch.tensor([True, False]), losses).mean()
    assert result.item() == pytest.approx((0.8 * 1.0 + 0.2 * 4.0) / 2)
    result.backward()
    torch.testing.assert_close(
        losses.grad, torch.tensor([0.4, 0.1], dtype=torch.float64)
    )
    assert probability.grad.item() == pytest.approx(-1.5)


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_change_head_uses_readout_and_belongs_to_optimizer(leading):
    policy = Policy(cfg_of(width=8), 2, ("a", "b")).double()
    with torch.no_grad():
        policy.change_head.weight.zero_()
        policy.change_head.weight[0, 0] = math.log(3)
    readout = torch.zeros((*leading, 8), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        readout[..., 0] = 1
    result = policy.change(readout)
    torch.testing.assert_close(result, torch.full(leading, 0.75, dtype=torch.float64))
    result.sum().backward()
    assert torch.count_nonzero(readout.grad)
    assert torch.count_nonzero(policy.change_head.weight.grad)
    assert any(p is policy.change_head.weight for p in policy.groups()["pi"])


def test_change_ce_counts_current_occurrences_and_weights_each_before_mean():
    logits = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64).log().requires_grad_()
    probability = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([2 / 3, 0, 1 / 3], dtype=torch.float64)
    result = change_cross_entropy(
        logits, q, probability, torch.tensor([True, True, False]), torch.tensor(True)
    )
    # Previous [A, B], current [A, A, C]: two repeated A losses, one changed C.
    assert result.item() == pytest.approx((1.6 * math.log(2) + 0.2 * math.log(5)) / 3)
    result.backward()
    weighted_q = torch.tensor([1.6 / 3, 0, 0.2 / 3], dtype=torch.float64)
    expected = (
        torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64) * weighted_q.sum()
        - weighted_q
    )
    torch.testing.assert_close(logits.grad, expected)
    assert probability.grad.item() == pytest.approx((2 * math.log(2) - math.log(5)) / 3)


@pytest.mark.parametrize("known", [False, True])
def test_change_ce_known_idle_and_unknown_predecessor(known):
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    probability = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    result = change_cross_entropy(
        logits,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        probability,
        torch.tensor([False, False]),
        torch.tensor(known),
    )
    assert result.item() == pytest.approx(0.2 * math.log(2) if known else 0)
    result.backward()
    assert probability.grad.item() == pytest.approx(-math.log(2) if known else 0)
    torch.testing.assert_close(
        logits.grad,
        torch.tensor([-0.1, 0.1] if known else [0.0, 0.0], dtype=torch.float64),
    )
