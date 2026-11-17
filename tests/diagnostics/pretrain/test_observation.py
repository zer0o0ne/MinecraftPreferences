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
