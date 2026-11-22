"""Standalone clip statistics remain independent of pair-only micro-batches."""

import math

import pytest
import torch

from mcpref.diagnostics.pretrain.diagnostics import (
    PhiObservations,
    phi_suite,
    policy_suite,
)
from mcpref.diagnostics.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
    preference_observation,
    raw_policy_observation,
    step_observation,
)


def pair_micro() -> MicroObservations:
    preference = preference_observation(
        (torch.tensor([0.25]), torch.tensor([-0.5]), [{"p": 0.75, "inc": 0}])
    )
    return MicroObservations(preference, None, None, 0)


def clip_micro() -> MicroObservations:
    policy, rows = policy_observation(
        (
            torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            {
                "action_valid": torch.tensor([[True, False]]),
                "shift_valid": torch.tensor([[True, True]]),
            },
            torch.tensor([2.0]),
        ),
        calibration_bins=2,
        copy_eps=0.2,
    )
    phi = phi_observation(
        {
            "changed": torch.tensor([[True, False, True]]),
            "d_alpha": torch.tensor([[1.0, 2.0, 3.0]]),
            "z_norm": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "hinge": torch.tensor([[0.0, 0.0, 1.0, 2.0]]),
        }
    )
    return MicroObservations(None, policy, phi, rows, {"marker": "clip"})


def test_pair_only_step_has_preferences_without_own_loss_observations():
    observed = step_observation([pair_micro(), pair_micro()], clips=0)
    assert observed.preference.p.tolist() == [0.75, 0.75]
    assert observed.policy is None
    assert observed.phi == PhiObservations(0, 0, 0, 0, 0, 0, 0)
    assert observed.raw_policy == ()
    policy = policy_suite(
        [observed.policy], calibration_bins=2, copy_eps=0.2, full=True
    )
    assert policy["action"]["count"] == 0
    assert policy["uniform"] is policy["copy_current"] is policy["zero_shift"] is None
    assert policy["empirical_constant_oracle"] is None
    phi = phi_suite([observed.phi])
    assert phi["d_alpha_mean"] is phi["z_norm_mean"] is phi["hinge_active"] is None


def test_mixed_step_normalizes_only_standalone_clips_and_keeps_all_pairs():
    observed = step_observation(
        [pair_micro(), clip_micro(), pair_micro(), clip_micro()], clips=2
    )
    assert observed.preference.p.tolist() == [0.75, 0.75]
    policy = observed.policy
    assert policy.action_valid == 2 and policy.shift_valid == 4
    assert policy.q_sum.tolist() == [0.5, 0.0]
    assert policy.uniform == pytest.approx(math.log(2) / 2)
    assert policy.copy_current == pytest.approx(-math.log(0.9) / 2)
    assert policy.zero_shift == 2.0
    assert policy.calibration_count.tolist() == [0, 2]
    assert observed.phi == PhiObservations(12, 8, 4, 6, 20, 8, 4)
    assert observed.raw_policy == ({"marker": "clip"}, {"marker": "clip"})
    suite = policy_suite(
        [None, policy, None], calibration_bins=2, copy_eps=0.2, full=True
    )
    assert suite["action"]["count"] == 2
    assert suite["action"]["top1_support"] == 1
    assert suite["uniform"] == policy.uniform


@pytest.mark.parametrize("copy_eps", [0.0, 0.2])
def test_copy_without_previous_action_is_uniform(copy_eps):
    value = (
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64),
        {"action_valid": torch.tensor([[True]]), "shift_valid": torch.tensor([[True]])},
        torch.tensor([0.0]),
    )
    result, _ = policy_observation(value, calibration_bins=2, copy_eps=copy_eps)
    assert result["copy_finite"]
    assert result["copy_current"].item() == pytest.approx(math.log(2))


def test_raw_policy_rows_use_source_masks_and_interval_actions(tmp_path):
    from models.policy.policy_world import action_masks, make_clip

    from mcpref.models.policy.targets import targets, weights

    masks = action_masks(tmp_path / "masks")
    index = {name: i for i, name in enumerate(masks.names)}
    clip = make_clip([["hotbar_3"], ["turn_up"]], [0.0, -90.0, 0.0], [0, 2, 0])
    clip.info["relpath"] = "episode"
    q, chi = targets(clip, masks)
    value = (
        torch.zeros(1, 3, len(index)),
        torch.tensor(q[None]),
        torch.tensor(weights(clip, index)[None, :-1]),
        {
            "action_valid": torch.tensor(q.sum(-1)[None] > 0),
            "shift_valid": torch.tensor(chi[None]),
        },
        torch.zeros(1),
    )
    got = raw_policy_observation(value, [clip], masks)
    assert got["actions"].tolist() == [index["hotbar_3"], index["turn_up"]]
    assert got["masks"].shape == (2, len(index))
    assert got["masks"][:, index["hotbar_3"]].tolist() == [True, False]
    assert got["masks"][:, index["turn_up"]].tolist() == [True, False]
    assert got["previous"][0].sum() == 0
    assert got["previous"][1, index["hotbar_3"]] == 1
    assert got["clusters"] == ("episode", "episode")
