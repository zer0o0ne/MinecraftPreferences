"""Existing gradient and optimizer-update health observations."""

import math

import torch

from mcpref.diagnostics.core.config import DiagnosticsConfig
from mcpref.diagnostics.core.observations import UpdateObservations
from mcpref.diagnostics.core.stats import distribution, rate
from mcpref.diagnostics.core.types import ProducerContext, metric_record


def _norm(value: torch.Tensor) -> float:
    dtype = torch.promote_types(value.dtype, torch.float32)
    return float(torch.linalg.vector_norm(value.detach(), ord=2, dtype=dtype).cpu())


def _scalar(value: float | None, cfg: DiagnosticsConfig, reason: str | None = None):
    return distribution(
        [] if value is None else [value],
        cfg,
        missing=int(value is None),
        reason=reason if value is None else None,
    )


def on_update_observation(
    observation: UpdateObservations, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(18): norms and support of already computed gradients and updates."""
    metrics, populations = {}, {}
    cluster = list(context.cluster_ids)
    group_norms = {}
    for group, names in observation.groups.items():
        tensors = [
            observation.gradients[name]
            for name in names
            if name in observation.gradients
        ]
        norm = (
            math.sqrt(sum(_norm(value) ** 2 for value in tensors)) if tensors else None
        )
        group_norms[group] = norm
        metrics[f"update.group.{group}.preclip_norm"] = _scalar(norm, cfg)
        metrics[f"update.group.{group}.nonfinite"] = sum(
            int((~torch.isfinite(observation.gradients[name])).sum())
            for name in names
            if name in observation.gradients
        )
        metrics[f"update.group.{group}.active"] = int(bool(tensors))
        update = [observation.updates[name] for name in names]
        unavailable = any(value is None for value in update)
        update_norm = (
            None
            if unavailable
            else math.sqrt(sum(_norm(value) ** 2 for value in update))
        )
        metrics[f"update.group.{group}.norm"] = _scalar(
            update_norm, cfg, "zero_decay_factor"
        )
    total = math.sqrt(
        sum(value**2 for value in group_norms.values() if value is not None)
    )
    metrics["update.preclip_norm"] = _scalar(
        total if any(value is not None for value in group_norms.values()) else None, cfg
    )
    metrics["update.clip_coefficient"] = _scalar(observation.clip_coefficient, cfg)
    metrics["update.learning_rate"] = _scalar(observation.learning_rate, cfg)
    metrics["update.schedule_position"] = observation.schedule_position
    logits = observation.logits.detach().to(torch.float64)
    logit_norm = torch.linalg.vector_norm(logits, ord=2, dim=-1).cpu().numpy()
    metrics["update.logit_norm"] = distribution(logit_norm, cfg)
    head_gradient = observation.gradients.get("pi.head.weight")
    head_update = observation.updates.get("pi.head.weight")
    if head_gradient is not None:
        rows = (
            torch.linalg.vector_norm(
                head_gradient.detach().to(torch.float64), ord=2, dim=-1
            )
            .cpu()
            .numpy()
        )
        metrics["update.head_row_norm"] = distribution(rows, cfg)
        zero = rows == 0
        supported = observation.supported_actions.detach().cpu().numpy().astype(bool)
        unsupported = ~supported
        metrics["update.zero_gradient_row"] = rate(
            zero.astype(float).tolist(), cluster, cfg
        )
        metrics["update.unsupported_action"] = rate(
            unsupported.astype(float).tolist(), cluster, cfg
        )
        metrics["update.zero_unsupported_row"] = rate(
            (zero & unsupported).astype(float).tolist(), cluster, cfg
        )
    if head_update is None:
        metrics["update.unavailable"] = _scalar(None, cfg, "zero_decay_factor")
    else:
        metrics["update.head_update_row_norm"] = distribution(
            torch.linalg.vector_norm(
                head_update.detach().to(torch.float64), ord=2, dim=-1
            )
            .cpu()
            .numpy(),
            cfg,
        )
    populations = {
        name: cluster
        for name, value in metrics.items()
        if isinstance(value, dict) and "n" in value
    }
    step = context.steps[0] if context.steps else observation.schedule_position
    return metric_record(
        "rl", f"{context.producer}:{step}", context, metrics, populations
    )
