"""Supervised policy, mask, PPO and factual/METRA observations."""

import math

import numpy as np
import torch

from mcpref.diagnostics.core.config import DiagnosticsConfig
from mcpref.diagnostics.core.observations import (
    PolicyObservations,
    PpoObservations,
    ShiftObservations,
)
from mcpref.diagnostics.core.stats import distribution, rate
from mcpref.diagnostics.core.types import ProducerContext, metric_record


def _array(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float64).numpy()


def _entropy(rows: np.ndarray) -> np.ndarray:
    return -np.sum(
        np.where(rows > 0, rows * np.log(np.where(rows > 0, rows, 1)), 0), axis=1
    )


def _cross_entropy(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    valid = (target > 0) & (probability > 0)
    result = np.full(len(target), np.nan)
    defined = np.all((target <= 0) | (probability > 0), axis=1)
    terms = np.where(valid, target * np.log(np.where(valid, probability, 1)), 0)
    result[defined] = -np.sum(terms[defined], axis=1)
    return result


def _finite(values: np.ndarray) -> tuple[list[float], int, str | None]:
    finite = np.isfinite(values)
    return (
        values[finite].tolist(),
        int((~finite).sum()),
        None if finite.all() else "undefined_log",
    )


def _topk(row: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-row, kind="stable")[:k]


def _record(context: ProducerContext, metrics: dict, populations: dict, joined=None):
    step = context.steps[0] if context.steps else 0
    return metric_record(
        "rl", f"{context.producer}:{step}", context, metrics, populations
    ) | {"joined": joined or {}}


def _calibration(probability, target, valid, bins):
    confidence = probability.max(axis=1)
    accuracy = (
        np.take_along_axis(target, probability.argmax(axis=1)[:, None], axis=1)[:, 0]
        > 0
    )
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = valid & (indices == index)
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else None
        mean_accuracy = float(accuracy[selected].mean()) if count else None
        if count:
            ece += count * abs(mean_confidence - mean_accuracy)
        rows.append(
            {
                "bin": index,
                "low": edges[index],
                "high": edges[index + 1],
                "n": count,
                "confidence": mean_confidence,
                "accuracy": mean_accuracy,
            }
        )
    return confidence, ece / max(int(valid.sum()), 1), rows


def on_policy_observation(
    observation: PolicyObservations, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(8)-(12): full-softmax quality, baselines, masks and calibration."""
    probability = torch.softmax(observation.logits, dim=-1, dtype=None)
    pi, q = _array(probability), _array(observation.targets)
    mask = _array(observation.masks).astype(bool)
    previous, prior = _array(observation.previous), _array(observation.prior)
    actions = _array(observation.actions).astype(int)
    if not (
        len(pi)
        == len(q)
        == len(mask)
        == len(previous)
        == len(actions)
        == len(context.cluster_ids)
    ):
        raise ValueError("policy observations and cluster ids must be aligned")
    valid = q.sum(axis=1) > 0
    singleton = (q == 1).sum(axis=1) == 1
    clusters = np.asarray(context.cluster_ids)
    vc = clusters[valid].tolist()
    top1 = np.take_along_axis(q, pi.argmax(axis=1)[:, None], axis=1)[:, 0] > 0
    topk = np.zeros((len(q), pi.shape[1]), dtype=bool)
    for i, row in enumerate(pi):
        topk[i, _topk(row, observation.width)] = True
    nll, missing, reason = _finite(_cross_entropy(q, pi)[valid])
    kl_teacher = _cross_entropy(q, pi) - _entropy(q)
    kl_model = _cross_entropy(pi, q) - _entropy(pi)
    prior_ce = _cross_entropy(q, np.broadcast_to(prior, q.shape))
    copy = (1 - observation.copy_eps) * previous + observation.copy_eps / pi.shape[1]
    metrics = {
        "policy.nll": distribution(nll, cfg, missing=missing, reason=reason),
        "policy.brier": distribution(np.sum((pi - q) ** 2, axis=1)[valid], cfg),
        "policy.top1_support": rate(top1[valid].astype(float).tolist(), vc, cfg),
        "policy.topk_support": rate(
            ((topk & (q > 0)).sum(axis=1) > 0)[valid].astype(float).tolist(), vc, cfg
        ),
        "policy.topk_target_mass": distribution((topk * q).sum(axis=1)[valid], cfg),
        "policy.entropy": distribution(_entropy(pi)[valid], cfg),
        "policy.teacher_entropy": distribution(_entropy(q)[valid], cfg),
        "policy.tv": distribution((np.abs(pi - q).sum(axis=1) / 2)[valid], cfg),
    }
    for name, values in (
        ("kl_teacher", kl_teacher),
        ("kl_model", kl_model),
        ("prior_ce", prior_ce),
        ("copy_ce", _cross_entropy(q, copy)),
    ):
        finite, absent, why = _finite(values[valid])
        metrics[f"policy.{name}"] = distribution(
            finite, cfg, missing=absent, reason=why
        )
    allowed = mask.sum(axis=1)
    uniform = np.divide(
        mask, allowed[:, None], out=np.zeros_like(pi), where=allowed[:, None] > 0
    )
    finite, absent, why = _finite(_cross_entropy(q, uniform)[valid])
    metrics["policy.uniform_allowed_ce"] = distribution(
        finite, cfg, missing=absent, reason=why
    )
    counts = mask.sum(axis=1)
    metrics |= {
        "mask.count": distribution(counts, cfg),
        "mask.empty": rate(
            (counts == 0).astype(float).tolist(), list(context.cluster_ids), cfg
        ),
        "mask.singleton": rate(
            (counts == 1).astype(float).tolist(), list(context.cluster_ids), cfg
        ),
        "mask.probability": distribution(1 - (pi * mask).sum(axis=1), cfg),
        "mask.executed_violation": rate(
            (~mask[np.arange(len(actions)), actions]).astype(float).tolist(),
            list(context.cluster_ids),
            cfg,
        ),
        "policy.singleton_teacher": rate(
            singleton.astype(float).tolist(), list(context.cluster_ids), cfg
        ),
        "policy.zero_teacher": rate(
            (~valid).astype(float).tolist(), list(context.cluster_ids), cfg
        ),
    }
    z = (pi * mask).sum(axis=1)
    normalized = np.divide(
        pi * mask, z[:, None], out=np.zeros_like(pi), where=z[:, None] > 0
    )
    entropy = _entropy(normalized)
    metrics["mask.allowed_entropy"] = distribution(
        entropy[z > 0],
        cfg,
        missing=int((z == 0).sum()),
        reason="zero_allowed_probability" if (z == 0).any() else None,
    )
    confidence, ece, reliability = _calibration(
        pi, q, valid, observation.calibration_bins
    )
    metrics["policy.confidence"] = distribution(confidence[valid], cfg)
    metrics["policy.ece"] = distribution([ece] if valid.any() else [], cfg)
    populations = {
        name: vc for name in metrics if name.startswith(("policy.", "mask."))
    }
    for name in (
        "mask.count",
        "mask.empty",
        "mask.singleton",
        "mask.probability",
        "mask.executed_violation",
        "policy.singleton_teacher",
        "policy.zero_teacher",
        "mask.allowed_entropy",
    ):
        populations[name] = (
            list(context.cluster_ids)
            if name != "mask.allowed_entropy"
            else clusters[z > 0].tolist()
        )
    confusion = np.zeros((pi.shape[1], pi.shape[1]), dtype=int)
    rows = valid & singleton
    for truth, predicted in zip(q[rows].argmax(axis=1), pi[rows].argmax(axis=1)):
        confusion[truth, predicted] += 1
    true_support, predicted_support = confusion.sum(axis=1), confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(
        diagonal,
        true_support,
        out=np.full(len(diagonal), np.nan),
        where=true_support > 0,
    )
    precision = np.divide(
        diagonal,
        predicted_support,
        out=np.full(len(diagonal), np.nan),
        where=predicted_support > 0,
    )
    metrics["policy.micro_recall"] = distribution(
        [diagonal.sum() / rows.sum()] if rows.any() else [], cfg
    )
    for name, values, support in (
        ("macro_recall", recall, true_support),
        ("macro_precision", precision, predicted_support),
    ):
        selected = support > 0
        metrics[f"policy.{name}"] = distribution(
            [float(np.nanmean(values))] if selected.any() else [], cfg
        )
    for name in ("micro_recall", "macro_recall", "macro_precision"):
        populations[f"policy.{name}"] = clusters[rows].tolist()
    return _record(
        context,
        metrics,
        populations,
        {
            "confusion": confusion.tolist(),
            "reliability": reliability,
            "recall_classes": np.flatnonzero(true_support > 0).tolist(),
            "precision_classes": np.flatnonzero(predicted_support > 0).tolist(),
        },
    )


def on_ppo_observation(
    observation: PpoObservations, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(12): stored PPO ratios, support-sensitive KL and actual populations."""
    rho, advantage = _array(observation.ratios), _array(observation.advantages)
    mu, pi = _array(observation.behavior), _array(observation.policy)
    actions = _array(observation.actions).astype(int)
    rows = np.arange(len(actions))
    chosen_mu, chosen_pi = mu[rows, actions], pi[rows, actions]
    if np.any(chosen_mu <= 0) or np.any(chosen_pi <= 0):
        raise ValueError("chosen PPO probabilities must be positive")
    clipped = np.clip(rho, 1 - observation.clip, 1 + observation.clip)
    metrics = {
        "ppo.ratio": distribution(rho, cfg),
        "ppo.ratio_max": distribution([rho.max()] if len(rho) else [], cfg),
        "ppo.advantage": distribution(advantage, cfg),
        "ppo.advantage_std": distribution(
            [advantage.std()] if len(advantage) else [], cfg
        ),
        "ppo.kl_sample": distribution(np.log(chosen_mu) - np.log(chosen_pi), cfg),
        "ppo.logprob_drift": distribution(
            np.abs(np.log(chosen_mu) - np.log(chosen_pi)), cfg
        ),
        "ppo.action_tv": distribution(np.abs(pi - mu).sum(axis=1) / 2, cfg),
    }
    for name, selected in (
        ("clip_positive", advantage > 0),
        ("clip_negative", advantage < 0),
    ):
        values = (rho * advantage != clipped * advantage)[selected]
        metrics[f"ppo.{name}"] = rate(
            values.astype(float).tolist(),
            np.asarray(context.cluster_ids)[selected].tolist(),
            cfg,
        )
    exact = []
    for behavior, policy in zip(mu, pi):
        exact.append(
            float(
                np.sum(
                    behavior[behavior > 0]
                    * (np.log(behavior[behavior > 0]) - np.log(policy[behavior > 0]))
                )
            )
            if np.all(policy[behavior > 0] > 0)
            else math.nan
        )
    finite, missing, reason = _finite(np.asarray(exact))
    metrics["ppo.kl_exact"] = distribution(finite, cfg, missing=missing, reason=reason)
    metrics["ppo.kl_exact_undefined"] = missing
    denominator = np.sum(rho**2)
    ess = np.sum(rho) ** 2 / denominator if denominator else None
    metrics["ppo.ess"] = distribution(
        [] if ess is None else [ess],
        cfg,
        missing=int(ess is None),
        reason="zero_ratio_squares" if ess is None else None,
    )
    metrics["ppo.ess_normalized"] = distribution(
        [] if ess is None else [ess / len(rho)],
        cfg,
        missing=int(ess is None),
        reason="zero_ratio_squares" if ess is None else None,
    )
    eligible = _array(observation.eligible).astype(bool)
    for name, selected in (
        ("eligible", eligible),
        ("fresh", np.asarray(observation.kinds) == "fresh"),
        ("pending", np.asarray(observation.kinds) == "pending"),
    ):
        metrics[f"ppo.{name}"] = rate(
            selected.astype(float).tolist(), list(context.cluster_ids), cfg
        )
        metrics[f"ppo.{name}_count"] = int(selected.sum())
    metrics["ppo.zero_advantage_count"] = int((advantage == 0).sum())
    populations = {
        name: list(context.cluster_ids)
        for name in metrics
        if not name.endswith("_count")
    }
    for name, selected in (("positive", advantage > 0), ("negative", advantage < 0)):
        populations[f"ppo.clip_{name}"] = np.asarray(context.cluster_ids)[
            selected
        ].tolist()
    populations["ppo.kl_exact"] = np.asarray(context.cluster_ids)[
        np.isfinite(exact)
    ].tolist()
    return _record(context, metrics, populations)


def on_shift_observation(
    observation: ShiftObservations, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(15): detached factual shift and existing METRA observations."""
    error = _array(observation.predicted) - _array(observation.actual)
    alpha = _array(observation.alpha)
    distance = np.sqrt(np.sum((error * alpha) ** 2, axis=-1))
    baseline = np.sqrt(np.sum((_array(observation.actual) * alpha) ** 2, axis=-1))
    metrics = {
        "shift.error": distribution(distance, cfg),
        "shift.error_squared": distribution(distance**2, cfg),
        "shift.zero_baseline": distribution(baseline, cfg),
        "metra.d": distribution(_array(observation.distance), cfg),
        "metra.q": distribution(_array(observation.constraint), cfg),
        "metra.lambda": distribution(_array(observation.dual), cfg),
        "metra.constraint_active": rate(
            (1 - _array(observation.distance) ** 2 <= observation.slack)
            .astype(float)
            .tolist(),
            list(context.cluster_ids),
            cfg,
        ),
        "metra.zero_skill": rate(
            _array(observation.zero_skill).astype(float).tolist(),
            list(context.cluster_ids),
            cfg,
        ),
    }
    return _record(
        context, metrics, {name: list(context.cluster_ids) for name in metrics}
    )
