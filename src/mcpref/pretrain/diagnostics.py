"""CPU observation contracts and pure suites of pretrain diagnostics."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mcpref.diagnostics.preference import on_pretrain_step, on_reference
from mcpref.diagnostics.types import ReferencePair
from mcpref.preferences.diagnostics import (
    BASELINE_KEYS,
    GlobalFit,
    PreferenceRows,
    Stratum,
    preference_suite,
    stratum_suite,
)
from mcpref.preferences.loss import (
    preference_baselines,
    probability_ce,
    weighted_ce,
)
from mcpref.preferences.metrics import (
    calibration,
    calibration_bins,
    head_suite,
    inc_classification,
    mean_nullable,
    quantile,
)

__all__ = [
    "BASELINE_KEYS",
    "GlobalFit",
    "PhiObservations",
    "PolicyObservations",
    "PreferenceRows",
    "ReferencePair",
    "StepObservations",
    "Stratum",
    "calibration",
    "head_suite",
    "inc_classification",
    "mean_nullable",
    "on_pretrain_step",
    "on_reference",
    "phi_suite",
    "policy_suite",
    "preference_baselines",
    "preference_suite",
    "probability_ce",
    "quantile",
    "stratum_suite",
    "weighted_ce",
]


@dataclass(frozen=True)
class PolicyObservations:
    q_sum: torch.Tensor
    action_valid: int
    shift_valid: int
    confidence_sum: float
    correct_sum: int
    entropy_sum: float
    calibration_count: torch.Tensor
    calibration_confidence: torch.Tensor
    calibration_correct: torch.Tensor
    uniform: float
    copy_current: float | None
    zero_shift: float


@dataclass(frozen=True)
class PhiObservations:
    d_sum: float
    d_changed_sum: float
    changed: int
    state_pairs: int
    z_norm_sum: float
    z_count: int
    hinge_active: int


@dataclass(frozen=True)
class StepObservations:
    preference: PreferenceRows | None
    policy: PolicyObservations
    phi: PhiObservations
    raw_policy: tuple = ()


def action_suite(
    observations: Sequence[PolicyObservations], calibration_bins_count: int, full: bool
) -> dict:
    """(16): pooled valid action-row observations."""
    count = sum(observation.action_valid for observation in observations)
    confidence = sum(observation.confidence_sum for observation in observations)
    correct = sum(observation.correct_sum for observation in observations)
    entropy = sum(observation.entropy_sum for observation in observations)
    result = {
        "count": count,
        "top1_support": correct / count if count else None,
        "mean_confidence": confidence / count if count else None,
        "action_entropy": entropy / count if count else None,
    }
    if full:
        count_sums = sum(
            (observation.calibration_count for observation in observations),
            start=torch.zeros(calibration_bins_count, dtype=torch.int64),
        )
        confidence_sums = sum(
            (observation.calibration_confidence for observation in observations),
            start=torch.zeros(calibration_bins_count, dtype=torch.float64),
        )
        correct_sums = sum(
            (observation.calibration_correct for observation in observations),
            start=torch.zeros(calibration_bins_count, dtype=torch.float64),
        )
        bins, ece = calibration_bins(
            count_sums.tolist(),
            confidence_sums.tolist(),
            correct_sums.tolist(),
            calibration_bins_count,
        )
        result |= {"calibration": bins, "ece": ece}
    return result


def policy_suite(
    observations: Sequence[PolicyObservations],
    *,
    calibration_bins: int,
    copy_eps: float,
    full: bool,
) -> dict:
    """(16): step action rows or a pooled policy suite."""
    action = action_suite(observations, calibration_bins, full)
    if not full:
        return action
    if not observations:
        return {
            "action": action,
            "empirical_constant_oracle": None,
            "uniform": None,
            "copy_current": None,
            "zero_shift": None,
        }
    q_sum = sum(
        (observation.q_sum for observation in observations),
        start=torch.zeros_like(observations[0].q_sum),
    )
    total = q_sum.sum().item()
    oracle = (
        -sum(
            weight * math.log(weight / total) for weight in q_sum.tolist() if weight > 0
        )
        / total
        if total
        else None
    )
    uniform, _ = mean_nullable(observation.uniform for observation in observations)
    copy_current, _ = mean_nullable(
        observation.copy_current for observation in observations
    )
    zero_shift = mean_nullable(observation.zero_shift for observation in observations)[
        0
    ]
    return {
        "action": action,
        "empirical_constant_oracle": oracle,
        "uniform": uniform,
        "copy_current": copy_current,
        "zero_shift": zero_shift,
    }


def phi_suite(observations: Sequence[PhiObservations]) -> dict[str, float | int | None]:
    """(4): pooled perception observations."""
    pairs = sum(observation.state_pairs for observation in observations)
    changed = sum(observation.changed for observation in observations)
    z_count = sum(observation.z_count for observation in observations)
    return {
        "d_alpha_mean": (
            sum(observation.d_sum for observation in observations) / pairs
            if pairs
            else None
        ),
        "d_alpha_changed_mean": (
            sum(observation.d_changed_sum for observation in observations) / changed
            if changed
            else None
        ),
        "z_norm_mean": (
            sum(observation.z_norm_sum for observation in observations) / z_count
            if z_count
            else None
        ),
        "hinge_active": (
            sum(observation.hinge_active for observation in observations) / z_count
            if z_count
            else None
        ),
        "changed": changed,
        "state_pairs": pairs,
    }
