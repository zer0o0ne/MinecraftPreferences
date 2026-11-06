"""Detached micro-batch reductions of specs/pretrain-diagnostics.md (14)-(16)."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mcpref.preferences.diagnostics import pool_preferences
from mcpref.pretrain.diagnostics import (
    PhiObservations,
    PolicyObservations,
    PreferenceRows,
    StepObservations,
)


@dataclass(frozen=True)
class MicroObservations:
    preference: PreferenceRows | None
    policy: PolicyObservations
    phi: PhiObservations
    state_rows: int


def preference_observation(
    value: tuple[Tensor, Tensor, list[dict]] | None,
) -> PreferenceRows | None:
    if value is None:
        return None
    s, t, labels = value
    return PreferenceRows(
        s=s.detach().to("cpu", dtype=torch.float32),
        t=t.detach().to("cpu", dtype=torch.float32),
        p=torch.tensor(
            [label["p"] if label["inc"] == 0 else 0.5 for label in labels],
            dtype=torch.float32,
        ),
        inc=torch.tensor([label["inc"] for label in labels], dtype=torch.uint8),
    )


def action_calibration(
    confidence: Tensor, correct: Tensor, calibration_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    bins = torch.clamp(
        (confidence * calibration_bins).to(torch.int64), max=calibration_bins - 1
    ).cpu()
    count = torch.bincount(bins, minlength=calibration_bins)
    confidence_sum = torch.bincount(
        bins,
        weights=confidence.detach().cpu().to(torch.float64),
        minlength=calibration_bins,
    ).to(torch.float64)
    correct_sum = torch.bincount(
        bins,
        weights=correct.detach().cpu().to(torch.float64),
        minlength=calibration_bins,
    ).to(torch.float64)
    return count, confidence_sum, correct_sum


def action_statistics(
    rows: Tensor, targets: Tensor, valid_count: int, calibration_bins: int
) -> tuple[float, int, float, Tensor, Tensor, Tensor]:
    """(16): top-1, entropy and calibration sufficient statistics on valid rows."""
    actions = rows.shape[-1]
    probabilities = torch.softmax(rows, dim=-1, dtype=None)
    maximum = torch.amax(rows, dim=-1, keepdim=True)
    indices = torch.arange(actions, device=rows.device)
    top = torch.amin(torch.where(rows == maximum, indices, actions), dim=-1)
    confidence = torch.amax(probabilities, dim=-1)
    correct = targets[torch.arange(valid_count, device=rows.device), top] > 0
    positive = probabilities > 0
    safe_probabilities = torch.where(
        positive, probabilities, torch.ones_like(probabilities)
    )
    entropy = -torch.where(positive, probabilities * safe_probabilities.log(), 0.0).sum(
        dim=-1
    )
    calibration = action_calibration(confidence, correct, calibration_bins)
    return (
        confidence.sum().item(),
        int(correct.sum().item()),
        entropy.sum().item(),
        *calibration,
    )


def policy_observation(
    value: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor],
    calibration_bins: int,
    copy_eps: float,
) -> tuple[PolicyObservations, int]:
    logits, q, current, valid, zero_shift = value
    logits = logits[:, : q.shape[1]]
    action_valid, shift_valid = valid["action_valid"], valid["shift_valid"]
    actions = logits.shape[-1]
    valid_count = int(action_valid.sum().item())
    q_sum = q.detach().to(torch.float64).sum(dim=(0, 1)).cpu()
    r = (1 - copy_eps) * current[..., :actions] + copy_eps / actions
    infinite = bool(((q > 0) & (r == 0)).any().item())
    safe_r = torch.where(r > 0, r, torch.ones_like(r))
    copy_sum = -(q.to(torch.float64) * safe_r.to(torch.float64).log()).sum().item()
    rows, targets = logits[action_valid], q[action_valid]
    confidence, correct, entropy, count, confidence_sum, correct_sum = (
        action_statistics(rows, targets, valid_count, calibration_bins)
    )
    observation = PolicyObservations(
        q_sum=q_sum,
        action_valid=valid_count,
        shift_valid=int(shift_valid.sum().item()),
        confidence_sum=confidence,
        correct_sum=correct,
        entropy_sum=entropy,
        calibration_count=count,
        calibration_confidence=confidence_sum,
        calibration_correct=correct_sum,
        uniform=valid_count * math.log(actions),
        copy_current=None if infinite else copy_sum,
        zero_shift=zero_shift.sum().item(),
    )
    return observation, q.shape[1]


def phi_observation(diagnostics: dict[str, Tensor]) -> PhiObservations:
    changed = diagnostics["changed"]
    d_alpha = diagnostics["d_alpha"]
    z_norm = diagnostics["z_norm"]
    hinge = diagnostics["hinge"]
    return PhiObservations(
        d_sum=d_alpha.sum().item(),
        d_changed_sum=d_alpha[changed].sum().item(),
        changed=int(changed.sum().item()),
        state_pairs=changed.numel(),
        z_norm_sum=z_norm.sum().item(),
        z_count=z_norm.numel(),
        hinge_active=int((hinge > 0).sum().item()),
    )


def step_policy(micro: list[MicroObservations], clips: int) -> PolicyObservations:
    first = micro[0]
    denominator = clips * first.state_rows
    q_sum = (
        sum(
            (item.policy.q_sum for item in micro),
            start=torch.zeros_like(first.policy.q_sum),
        )
        / denominator
    )
    calibration = [
        sum(
            (getattr(item.policy, key) for item in micro),
            start=torch.zeros_like(getattr(first.policy, key)),
        )
        for key in (
            "calibration_count",
            "calibration_confidence",
            "calibration_correct",
        )
    ]
    copy_finite = all(item.policy.copy_current is not None for item in micro)
    return PolicyObservations(
        q_sum=q_sum,
        action_valid=sum(item.policy.action_valid for item in micro),
        shift_valid=sum(item.policy.shift_valid for item in micro),
        confidence_sum=sum(item.policy.confidence_sum for item in micro),
        correct_sum=sum(item.policy.correct_sum for item in micro),
        entropy_sum=sum(item.policy.entropy_sum for item in micro),
        calibration_count=calibration[0],
        calibration_confidence=calibration[1],
        calibration_correct=calibration[2],
        uniform=sum(item.policy.uniform for item in micro) / denominator,
        copy_current=(
            sum(item.policy.copy_current for item in micro) / denominator
            if copy_finite
            else None
        ),
        zero_shift=sum(item.policy.zero_shift for item in micro) / clips,
    )


def step_phi(micro: list[MicroObservations]) -> PhiObservations:
    return PhiObservations(
        **{
            key: sum(getattr(item.phi, key) for item in micro)
            for key in PhiObservations.__dataclass_fields__
        }
    )


def step_observation(micro: list[MicroObservations], clips: int) -> StepObservations:
    """Bounded per-step sufficient statistics; q_sum has the weights of (16)."""
    return StepObservations(
        pool_preferences(item.preference for item in micro),
        step_policy(micro, clips),
        step_phi(micro),
    )
