"""Detached micro-batch reductions of specs/pretrain-diagnostics.md (14)-(16)."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from mcpref.diagnostics.pretrain.diagnostics import (
    PhiObservations,
    PolicyObservations,
    PreferenceRows,
    StepObservations,
)
from mcpref.gui.mask import clip_masks
from mcpref.models.policy.targets import next_actions


@dataclass(frozen=True)
class MicroObservations:
    preference: PreferenceRows | None
    policy: dict[str, Tensor | int]
    phi: dict[str, Tensor | int]
    state_rows: int
    raw_policy: dict = field(default_factory=dict)


def preference_observation(
    value: tuple[Tensor, Tensor, list[dict]] | None, *, deferred: bool = False
) -> PreferenceRows | None:
    if value is None:
        return None
    s, t, labels = value
    if not deferred:
        s = s.detach().to("cpu", dtype=torch.float32)
        t = t.detach().to("cpu", dtype=torch.float32)
    return PreferenceRows(
        s=s.detach(),
        t=t.detach(),
        p=torch.tensor(
            [label["p"] if label["inc"] == 0 else 0.5 for label in labels],
            dtype=torch.float32,
            device="cpu",
        ),
        inc=torch.tensor(
            [label["inc"] for label in labels], dtype=torch.uint8, device="cpu"
        ),
    )


def action_calibration(
    confidence: Tensor, correct: Tensor, calibration_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    bins = torch.clamp(
        (confidence * calibration_bins).to(torch.int64), max=calibration_bins - 1
    )
    count = torch.bincount(bins, minlength=calibration_bins)
    confidence_sum = torch.bincount(
        bins,
        weights=confidence.detach().to(torch.float64),
        minlength=calibration_bins,
    ).to(torch.float64)
    correct_sum = torch.bincount(
        bins,
        weights=correct.detach().to(torch.float64),
        minlength=calibration_bins,
    ).to(torch.float64)
    return count, confidence_sum, correct_sum


def action_statistics(
    rows: Tensor, targets: Tensor, calibration_bins: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(16): top-1, entropy and calibration sufficient statistics on valid rows."""
    actions = rows.shape[-1]
    probabilities = torch.softmax(rows, dim=-1, dtype=None)
    maximum = torch.amax(rows, dim=-1, keepdim=True)
    indices = torch.arange(actions, device=rows.device)
    top = torch.amin(torch.where(rows == maximum, indices, actions), dim=-1)
    confidence = torch.amax(probabilities, dim=-1)
    correct = targets[torch.arange(rows.shape[0], device=rows.device), top] > 0
    positive = probabilities > 0
    safe_probabilities = torch.where(
        positive, probabilities, torch.ones_like(probabilities)
    )
    entropy = -torch.where(positive, probabilities * safe_probabilities.log(), 0.0).sum(
        dim=-1
    )
    calibration = action_calibration(confidence, correct, calibration_bins)
    return confidence.sum(), correct.sum(), entropy.sum(), *calibration


def policy_observation(
    value: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor],
    calibration_bins: int,
    copy_eps: float,
) -> tuple[dict[str, Tensor], int]:
    logits, q, current, valid, zero_shift = value
    logits = logits[:, : q.shape[1]]
    q, current = q.detach(), current.detach()
    action_valid, shift_valid = valid["action_valid"], valid["shift_valid"]
    actions = logits.shape[-1]
    valid_count = action_valid.sum()
    q_sum = q.to(torch.float64).sum(dim=(0, 1))
    r = (1 - copy_eps) * current[..., :actions] + copy_eps / actions
    copy_finite = ~((q > 0) & (r == 0)).any()
    safe_r = torch.where(r > 0, r, torch.ones_like(r))
    copy_sum = -(q.to(torch.float64) * safe_r.to(torch.float64).log()).sum()
    rows, targets = logits[action_valid], q[action_valid]
    confidence, correct, entropy, count, confidence_sum, correct_sum = (
        action_statistics(rows, targets, calibration_bins)
    )
    observation = {
        "q_sum": q_sum,
        "action_valid": valid_count,
        "shift_valid": shift_valid.sum(),
        "confidence_sum": confidence,
        "correct_sum": correct,
        "entropy_sum": entropy,
        "calibration_count": count,
        "calibration_confidence": confidence_sum,
        "calibration_correct": correct_sum,
        "uniform": valid_count.to(torch.float64) * math.log(actions),
        "copy_current": copy_sum,
        "copy_finite": copy_finite,
        "zero_shift": zero_shift.sum(),
    }
    return observation, q.shape[1]


def raw_policy_observation(value, clips, masks) -> dict:
    """Keep detached full rows for block 33 without another clip read or forward."""
    logits, q, current, valid, _ = value
    logits = logits[:, : q.shape[1]]
    actions_count = logits.shape[-1]
    if masks is None:
        allowed = np.ones(q.shape, dtype=bool)
        chosen = q.detach().argmax(dim=-1)
    else:
        index = {action: i for i, action in enumerate(masks.names)}
        allowed = np.stack(
            [clip_masks(masks, clip.info, clip.lines)[1:-1] for clip in clips]
        )
        chosen = np.stack([next_actions(clip, index)[:-2] for clip in clips])
    valid_rows = valid["action_valid"]
    prior = q[valid_rows].sum(dim=(0, 1))
    prior_sum = prior.sum()
    prior = torch.where(prior_sum > 0, prior / prior_sum, prior)
    return {
        "logits": logits.reshape(-1, actions_count).detach(),
        "targets": q.reshape(-1, actions_count).detach(),
        "previous": current[..., :actions_count].reshape(-1, actions_count).detach(),
        "masks": torch.tensor(allowed, dtype=torch.bool, device=logits.device).reshape(
            -1, actions_count
        ),
        "actions": torch.as_tensor(
            chosen, dtype=torch.int64, device=logits.device
        ).reshape(-1),
        "prior": prior.detach(),
        "clusters": tuple(
            clip.info["relpath"] for clip in clips for _ in range(q.shape[1])
        ),
    }


def phi_observation(diagnostics: dict[str, Tensor]) -> dict[str, Tensor | int]:
    changed = diagnostics["changed"]
    d_alpha = diagnostics["d_alpha"]
    z_norm = diagnostics["z_norm"]
    hinge = diagnostics["hinge"]
    return {
        "d_sum": d_alpha.sum(),
        "d_changed_sum": d_alpha[changed].sum(),
        "changed": changed.sum(),
        "state_pairs": changed.numel(),
        "z_norm_sum": z_norm.sum(),
        "z_count": z_norm.numel(),
        "hinge_active": (hinge > 0).sum(),
    }


POLICY_SCALARS = (
    "action_valid",
    "shift_valid",
    "confidence_sum",
    "correct_sum",
    "entropy_sum",
    "uniform",
    "copy_current",
    "copy_finite",
    "zero_shift",
)


def finish_policy(micro: list[MicroObservations]) -> list[PolicyObservations]:
    actions = micro[0].policy["q_sum"].numel()
    bins = micro[0].policy["calibration_count"].numel()
    packed = []
    for item in micro:
        policy = item.policy
        scalars = torch.stack([policy[key] for key in POLICY_SCALARS]).to(torch.float64)
        packed.append(
            torch.cat(
                [
                    scalars,
                    policy["q_sum"].to(torch.float64),
                    policy["calibration_count"].to(torch.float64),
                    policy["calibration_confidence"],
                    policy["calibration_correct"],
                ]
            )
        )
    rows = torch.stack(packed).cpu().tolist()
    result = []
    for values in rows:
        scalar = dict(zip(POLICY_SCALARS, values))
        offset = len(POLICY_SCALARS)
        q_sum = torch.tensor(
            values[offset : offset + actions], dtype=torch.float64, device="cpu"
        )
        offset += actions
        count = torch.tensor(
            values[offset : offset + bins], dtype=torch.int64, device="cpu"
        )
        offset += bins
        confidence = torch.tensor(
            values[offset : offset + bins], dtype=torch.float64, device="cpu"
        )
        offset += bins
        correct = torch.tensor(
            values[offset : offset + bins], dtype=torch.float64, device="cpu"
        )
        result.append(
            PolicyObservations(
                q_sum=q_sum,
                action_valid=int(scalar["action_valid"]),
                shift_valid=int(scalar["shift_valid"]),
                confidence_sum=scalar["confidence_sum"],
                correct_sum=int(scalar["correct_sum"]),
                entropy_sum=scalar["entropy_sum"],
                calibration_count=count,
                calibration_confidence=confidence,
                calibration_correct=correct,
                uniform=scalar["uniform"],
                copy_current=scalar["copy_current"] if scalar["copy_finite"] else None,
                zero_shift=scalar["zero_shift"],
            )
        )
    return result


def finish_phi(micro: list[MicroObservations]) -> list[PhiObservations]:
    keys = ("d_sum", "d_changed_sum", "changed", "z_norm_sum", "hinge_active")
    rows = (
        torch.stack([torch.stack([item.phi[key] for key in keys]) for item in micro])
        .cpu()
        .tolist()
    )
    return [
        PhiObservations(
            d_sum=values[0],
            d_changed_sum=values[1],
            changed=int(values[2]),
            state_pairs=item.phi["state_pairs"],
            z_norm_sum=values[3],
            z_count=item.phi["z_count"],
            hinge_active=int(values[4]),
        )
        for item, values in zip(micro, rows)
    ]


def step_preferences(micro: list[MicroObservations]) -> PreferenceRows | None:
    found = [item.preference for item in micro if item.preference is not None]
    if not found:
        return None
    s, t = (
        torch.stack(
            (torch.cat([row.s for row in found]), torch.cat([row.t for row in found]))
        )
        .cpu()
        .tolist()
    )
    return PreferenceRows(
        torch.tensor(s, dtype=torch.float32, device="cpu"),
        torch.tensor(t, dtype=torch.float32, device="cpu"),
        torch.cat([row.p for row in found]),
        torch.cat([row.inc for row in found]),
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
    preference = step_preferences(micro)
    policies, phis = finish_policy(micro), finish_phi(micro)
    finished = [
        MicroObservations(None, policy, phi, item.state_rows, item.raw_policy)
        for item, policy, phi in zip(micro, policies, phis)
    ]
    return StepObservations(
        preference,
        step_policy(finished, clips),
        step_phi(finished),
        tuple(item.raw_policy for item in micro),
    )
