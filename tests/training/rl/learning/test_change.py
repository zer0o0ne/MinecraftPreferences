"""Signed PPO and change-head gradients have independent hand-derived oracles."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mcpref.runtime.mcts.tree import LatentHistory
from mcpref.training.rl.learning import loss
from training.rl.learning.loss_fakes import modules, ppo_config

F64 = torch.float64


class Policy(nn.Module):
    names = ("a", "b")

    def __init__(self, probability):
        super().__init__()
        self.action_logits = nn.Parameter(
            torch.tensor([math.log(probability / (1 - probability)), 0.0], dtype=F64)
        )
        self.change_logit = nn.Parameter(torch.tensor(math.log(4.0), dtype=F64))

    def readout(self, y, c):
        return y

    def logits(self, readout):
        residual = torch.stack((readout[:, 0] - 0.2, readout[:, 0] * 0), dim=-1)
        return self.action_logits[None] + residual

    def change(self, readout):
        return (self.change_logit + readout[:, 1] + 0.3).sigmoid()

    def shift(self, readout, actions):
        return readout * 0


def setup(
    monkeypatch, probability, previous, *, kind="pending", advantage=1.0, frozen=False
):
    mods = modules()
    mods["pi"] = Policy(probability)
    mods["phi"].requires_grad_(not frozen)
    y = torch.tensor([[[0.2, -0.3]]], dtype=F64) * mods["phi"].scale / 1.2
    c = torch.zeros(1, 1, 3, dtype=F64)
    c[0, 0, 2 if previous is None else previous] = 1
    origin, successor = LatentHistory(y, c), LatentHistory(y + 0.1, c)
    monkeypatch.setattr(
        loss,
        "micro_histories",
        lambda *a, **kw: (
            iter([origin, successor] if kind == "fresh" else [origin]),
            iter([]),
        ),
    )
    monkeypatch.setattr(
        loss,
        "load_step",
        lambda *args: SimpleNamespace(
            action=0, probabilities=torch.tensor([0.5, 0.5], dtype=F64)
        ),
    )
    item = {"kind": kind, "index": 0 if kind == "fresh" else 2}
    args = {
        "sampling": SimpleNamespace(steps=4, out_dir="unused"),
        "perception": SimpleNamespace(alpha=(1.5, 0.5)),
        "states": 2,
        "alpha": torch.tensor([1.5, 0.5], dtype=F64),
        "ppo": ppo_config(horizon=1),
        "discount": 0.8,
        "min_p": 0.1,
        "fixed_advantages": {(kind, item["index"]): advantage},
    }
    return mods, item, args


@pytest.mark.parametrize("kind", ["fresh", "pending"])
@pytest.mark.parametrize("previous", [0, 1])
@pytest.mark.parametrize(
    "probability,advantage",
    [
        (0.5, 1.0),
        (0.5, -1.0),
        (0.5, 0.0),
        (0.8, 1.0),
        (0.2, -1.0),
        (0.2, 1.0),
        (0.8, -1.0),
    ],
)
def test_change_preserves_ppo_direction_and_trains_change_with_magnitude(
    monkeypatch, kind, previous, probability, advantage
):
    mods, item, args = setup(
        monkeypatch, probability, previous, kind=kind, advantage=advantage
    )
    terms, logged = loss.micro_loss(
        [item],
        mods,
        **args,
        metric=lambda delta: torch.linalg.vector_norm(delta, dim=-1),
    )
    ratio, p = 2 * probability, 0.8
    clipped_ratio = min(max(ratio, 0.8), 1.2)
    objective = -min(ratio * advantage, clipped_ratio * advantage)
    error = p if previous == 0 else 1 - p
    assert terms["change"][0].item() == pytest.approx(error * abs(objective))
    assert logged["change"][0].item() == pytest.approx(error * abs(objective))
    assert len(terms["change"]) == len(terms["ppo"]) == 1
    assert len(terms["shift"]) == int(kind == "fresh")
    terms["change"][0].backward()
    clipped = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < 0.8)
    slope = 0.0 if clipped else -advantage
    action_gradient = error * slope * ratio * (1 - probability)
    expected = torch.tensor([action_gradient, -action_gradient], dtype=F64)
    torch.testing.assert_close(mods["pi"].action_logits.grad, expected)
    direction = 1 if previous == 0 else -1
    assert mods["pi"].change_logit.grad.item() == pytest.approx(
        abs(objective) * direction * p * (1 - p)
    )


def test_no_previous_action_has_zero_auxiliary_value_and_gradients(monkeypatch):
    mods, item, args = setup(monkeypatch, 0.5, None)
    terms, _ = loss.micro_loss(
        [item],
        mods,
        **args,
        metric=lambda delta: torch.linalg.vector_norm(delta, dim=-1),
    )
    terms["change"][0].backward()
    assert terms["change"][0].item() == 0
    assert torch.count_nonzero(mods["pi"].action_logits.grad) == 0
    assert mods["pi"].change_logit.grad.item() == 0


@pytest.mark.parametrize("weight", [0.0, 3.0])
@pytest.mark.parametrize("frozen", [False, True])
def test_separate_stage_weight_uses_ppo_population_and_fixed_advantage(
    monkeypatch, weight, frozen
):
    mods, item, args = setup(monkeypatch, 0.5, 0, frozen=frozen)
    result = loss.step_gradients(
        [item],
        mods,
        **args,
        m=4,
        n_a=5,
        n_delta=0,
        r_c=0,
        w_ppo=2.0,
        w_change=weight,
        w_delta=1.0,
        w_metra=1.0,
        w_P=1.0,
        micro_batch=1,
    )
    assert result["ppo"] == -1.0
    assert result["change"] == pytest.approx(0.8)
    assert result["loss"] == pytest.approx(4 / 5 * (-2.0 + weight * 0.8))
    action_gradient = -0.5 * (2.0 + weight * 0.8) * 4 / 5
    torch.testing.assert_close(
        mods["pi"].action_logits.grad,
        torch.tensor([action_gradient, -action_gradient], dtype=F64),
    )
    if weight:
        assert mods["pi"].change_logit.grad.item() == pytest.approx(
            4 / 5 * weight * 0.16
        )
    else:
        assert mods["pi"].change_logit.grad is None
    if frozen:
        assert mods["phi"].scale.grad is None
    else:
        expected_phi = (0.2 * action_gradient - 0.3 * 4 / 5 * weight * 0.16) / 1.2
        assert mods["phi"].scale.grad.item() == pytest.approx(expected_phi)
