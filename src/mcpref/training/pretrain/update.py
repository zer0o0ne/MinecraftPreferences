"""Detached gradient and optimizer-update observations."""

import math
from pathlib import Path

import torch
from torch import Tensor

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.training.groups import Groups
from mcpref.training.optim import iter_update_terms
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.gradients.work import dot, load_vector, work_manifests


def squared_norm(value: Tensor) -> float:
    dtype = torch.promote_types(value.dtype, torch.float32)
    return torch.linalg.vector_norm(value, ord=2, dtype=dtype).item() ** 2


def norm_activity(parameters: list[torch.nn.Parameter]) -> dict:
    gradients = [p.grad for p in parameters if p.grad is not None]
    active = bool(gradients)
    squared = sum(squared_norm(gradient.detach()) for gradient in gradients)
    return {"norm": math.sqrt(squared) if active else None, "active": active}


@measured("diagnostics.gradients")
def gradient_stats(
    groups: Groups, grad_clip: float, *, preference: torch.nn.Module | None
) -> dict:
    """(12) of specs/pretrain-diagnostics.md, before any clipping."""
    group_stats = {
        name: norm_activity(parameters) for name, parameters in groups.items()
    }
    active = any(value["active"] for value in group_stats.values())
    norm = (
        math.sqrt(
            sum(value["norm"] ** 2 for value in group_stats.values() if value["active"])
        )
        if active
        else None
    )
    global_stats = {
        "norm": norm,
        "clip_coefficient": min(1, grad_clip / (norm + 1e-6)) if active else None,
        "active": active,
    }
    heads = None
    if preference is not None:
        heads = {
            "pref_weight": norm_activity([preference.pref.weight]),
            "inc_weight": norm_activity([preference.inc.weight]),
            "inc_bias": norm_activity([preference.inc.bias]),
        }
    return {"global": global_stats, "groups": group_stats, "P_heads": heads}


def update_result(squared: dict) -> dict:
    result = {}
    for group, (delta_squared, before_squared, unavailable) in squared.items():
        norm = math.sqrt(delta_squared) if not unavailable else None
        denominator = math.sqrt(before_squared)
        result[group] = {
            "norm": norm,
            "relative": (
                norm / denominator if not unavailable and denominator != 0 else None
            ),
            "unavailable": "zero_decay_factor" if unavailable else None,
        }
    return {"groups": result}


@measured("diagnostics.updates")
def update_stats(
    cfg: PretrainConfig,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    lr: dict,
) -> dict:
    """(13): ordinary group update norms through the shared parameter iterator."""
    squared = {group: [0.0, 0.0, False] for group in groups}
    for term in iter_update_terms(cfg, optimizer, groups, lr):
        values = squared[term.group]
        if term.unavailable is not None:
            values[2] = True
            continue
        values[0] += squared_norm(term.delta)
        values[1] += squared_norm(term.before)
    return update_result(squared)


def update_and_align(
    cfg: PretrainConfig,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    lr: dict,
    work: Path,
    progress=None,
) -> tuple[dict, dict]:
    """(26): ordinary updates and preference/update dots in one parameter pass."""
    manifests = work_manifests(work)
    squared = {group: [0.0, 0.0, False] for group in groups}
    alignment = {"phi": {}, "P": {}}
    unavailable_groups = set()
    for term in iter_update_terms(cfg, optimizer, groups, lr):
        values = squared[term.group]
        if term.unavailable is not None:
            values[2] = True
            unavailable_groups.add(term.group)
        else:
            values[0] += squared_norm(term.delta)
            values[1] += squared_norm(term.before)
        x = "P" if term.parameter.startswith("P.") else "phi"
        if not term.parameter.startswith(("phi.", "phi_loss.", "P.")):
            continue
        vector = load_vector(work, manifests, x, "P", term.parameter, progress)
        unavailable = term.unavailable
        value = None if unavailable else dot(vector, term.delta.detach().cpu())
        if value is not None and not math.isfinite(value):
            value = None
        bundle = alignment[x].setdefault(
            term.group, {"update_dot": 0.0, "update_unavailable": None}
        )
        if unavailable is not None:
            bundle["update_dot"] = None
            bundle["update_unavailable"] = unavailable
        elif bundle["update_dot"] is not None:
            bundle["update_dot"] = (
                bundle["update_dot"] + value if value is not None else None
            )
    for bundles in alignment.values():
        for group, bundle in bundles.items():
            if group in unavailable_groups:
                bundle["update_dot"] = None
                bundle["update_unavailable"] = "zero_decay_factor"
    return update_result(squared), alignment
