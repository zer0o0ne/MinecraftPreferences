"""Independent hand oracles for policy, PPO and factual/METRA metrics."""

import math

import torch

from mcpref.diagnostics.observations import (
    PolicyObservations,
    PpoObservations,
    ShiftObservations,
)
from mcpref.diagnostics.policy import (
    on_policy_observation,
    on_ppo_observation,
    on_shift_observation,
)

from .world import config, context


def test_policy_soft_metrics_kl_null_masks_and_confusion():
    probabilities = torch.tensor([[0.8, 0.2], [0.5, 0.5], [1.0, 0.0]])
    observation = PolicyObservations(
        logits=probabilities.log(),
        targets=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]]),
        masks=torch.tensor([[True, True], [True, False], [False, False]]),
        previous=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        actions=torch.tensor([0, 1, 0]),
        prior=torch.tensor([0.75, 0.25]),
        width=1,
        copy_eps=0.1,
        calibration_bins=2,
    )
    ctx = context(
        "rl/checkpoints/000000/losses.json",
        "policy",
        steps=(1,),
        clusters=("e0", "e1", "e2"),
    )
    record = on_policy_observation(observation, cfg=config(), context=ctx)
    metrics = record["metrics"]
    assert math.isclose(
        metrics["policy.nll"]["mean"],
        (-math.log(0.8) + math.log(2)) / 2,
        abs_tol=1e-6,
    )
    assert math.isclose(metrics["policy.brier"]["mean"], 0.04, abs_tol=1e-6)
    assert metrics["policy.top1_support"]["value"] == 1.0
    assert metrics["policy.topk_support"]["value"] == 1.0
    assert math.isclose(metrics["policy.tv"]["mean"], 0.1, abs_tol=1e-6)
    assert (
        metrics["policy.kl_model"]["n"] == 1
        and metrics["policy.kl_model"]["missing"] == 1
    )
    assert metrics["policy.uniform_allowed_ce"]["n"] == 1
    assert math.isclose(metrics["policy.uniform_allowed_ce"]["mean"], math.log(2))
    assert metrics["mask.empty"]["value"] == 1 / 3
    assert metrics["mask.executed_violation"]["value"] == 2 / 3
    assert metrics["policy.zero_teacher"]["value"] == 1 / 3
    assert record["joined"]["confusion"] == [[1, 0], [0, 0]]
    assert metrics["policy.micro_recall"]["mean"] == 1.0
    assert record["populations"]["policy.micro_recall"] == ["e0"]
    assert metrics["policy.ece"]["n"] == 1


def test_ppo_clip_signs_ess_support_and_actual_populations():
    observation = PpoObservations(
        ratios=torch.tensor([2.0, 0.5, 0.0]),
        advantages=torch.tensor([1.0, -1.0, 0.0]),
        behavior=torch.tensor([[0.8, 0.2], [0.5, 0.5], [1.0, 0.0]]),
        policy=torch.tensor([[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]]),
        actions=torch.tensor([0, 1, 0]),
        clip=0.2,
        kinds=("fresh", "pending", "fresh"),
        eligible=torch.tensor([True, True, False]),
    )
    record = on_ppo_observation(
        observation,
        cfg=config(),
        context=context(
            "rl/checkpoints/000000/losses.json",
            "ppo",
            steps=(2,),
            clusters=("e0", "e0", "e1"),
        ),
    )
    metrics = record["metrics"]
    assert metrics["ppo.clip_positive"]["value"] == 1.0
    assert metrics["ppo.clip_negative"]["value"] == 1.0
    assert record["populations"]["ppo.clip_positive"] == ["e0"]
    assert record["populations"]["ppo.clip_negative"] == ["e0"]
    assert metrics["ppo.kl_exact"]["n"] == 3
    assert math.isclose(metrics["ppo.ess"]["mean"], 2.5**2 / 4.25)
    assert metrics["ppo.zero_advantage_count"] == 1
    assert metrics["ppo.fresh_count"] == 2 and metrics["ppo.pending_count"] == 1
    assert metrics["ppo.eligible"]["value"] == 2 / 3


def test_shift_and_metra_use_weighted_distance_without_gradients():
    observation = ShiftObservations(
        predicted=torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
        actual=torch.tensor([[1.0, 1.0], [0.0, 2.0]]),
        alpha=torch.tensor([2.0, 1.0]),
        distance=torch.tensor([0.5, 1.0]),
        constraint=torch.tensor([0.1, 0.0]),
        dual=torch.tensor([3.0, 3.0]),
        slack=0.01,
        zero_skill=torch.tensor([False, True]),
    )
    record = on_shift_observation(
        observation,
        cfg=config(),
        context=context(
            "rl/checkpoints/000000/losses.json",
            "shift",
            steps=(3,),
            clusters=("e0", "e1"),
        ),
    )
    metrics = record["metrics"]
    assert metrics["shift.error"]["mean"] == (math.sqrt(5) + 2) / 2
    assert metrics["shift.zero_baseline"]["mean"] == (math.sqrt(5) + 2) / 2
    assert metrics["metra.constraint_active"]["value"] == 0.5
    assert metrics["metra.zero_skill"]["value"] == 0.5
