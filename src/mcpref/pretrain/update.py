"""(4)-(6) of specs/pretrain.md and detached gradient/update observations."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from mcpref.pretrain.checks import Groups
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.gradient_work import dot, load_vector, work_manifests


def lr_factor(t: int, total: int, warmup: int, floor: float) -> float:
    """(5): f(t), linear warmup over T_w steps, then cosine decay to rho at T."""
    if t <= warmup:
        return t / warmup
    return floor + (1 - floor) * 0.5 * (
        1 + math.cos(math.pi * (t - warmup) / (total - warmup))
    )


def adamw(cfg: PretrainConfig, groups: Groups) -> torch.optim.AdamW:
    """(6): AdamW with one parameter group per group, its eta_q and lambda_q."""
    param_groups = [{"params": params} | cfg.groups[q] for q, params in groups.items()]
    return torch.optim.AdamW(
        param_groups, betas=cfg.betas, eps=cfg.adam_eps, amsgrad=False, maximize=False
    )


def update(
    cfg: PretrainConfig,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    t: int,
    total: int,
) -> dict:
    """(4)-(6) of step t: clipping over G, eta_{q,t} set, AdamW step; {q: eta_{q,t}}."""
    params = [p for g in optimizer.param_groups for p in g["params"]]
    torch.nn.utils.clip_grad_norm_(
        params, max_norm=cfg.grad_clip, norm_type=2.0, error_if_nonfinite=False
    )
    f, lr = lr_factor(t, total, cfg.warmup_steps, cfg.lr_floor), {}
    for g, q in zip(optimizer.param_groups, groups):
        g["lr"] = lr[q] = cfg.groups[q]["lr"] * f
    optimizer.step()
    return lr


def squared_norm(value: Tensor) -> float:
    return torch.linalg.vector_norm(value, ord=2).item() ** 2


def norm_activity(parameters: list[torch.nn.Parameter]) -> dict:
    gradients = [p.grad for p in parameters if p.grad is not None]
    active = bool(gradients)
    squared = sum(squared_norm(gradient.detach()) for gradient in gradients)
    return {"norm": math.sqrt(squared) if active else None, "active": active}


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


@dataclass(frozen=True)
class UpdateTerm:
    group: str
    parameter: str
    before: Tensor | None
    delta: Tensor | None
    unavailable: str | None


def iter_update_terms(
    cfg: PretrainConfig,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    lr: dict,
) -> Iterator[UpdateTerm]:
    """(13): reconstruct one parameter's update from its post-step AdamW state."""
    beta1, beta2 = cfg.betas
    for group, parameters in groups.items():
        decay = cfg.groups[group]["weight_decay"]
        factor = 1 - lr[group] * decay
        for parameter in parameters:
            name = groups.parameter_names[id(parameter)]
            if parameter.grad is None:
                yield UpdateTerm(
                    group, name, parameter.detach(), torch.zeros_like(parameter), None
                )
                continue
            if factor == 0:
                yield UpdateTerm(group, name, None, None, "zero_decay_factor")
                continue
            state = optimizer.state[parameter]
            step = int(state["step"].item())
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            update = (state["exp_avg"] / correction1) / (
                state["exp_avg_sq"].sqrt() / math.sqrt(correction2) + cfg.adam_eps
            )
            delta = -(lr[group] / factor) * (decay * parameter.detach() + update)
            yield UpdateTerm(group, name, parameter.detach() - delta, delta, None)


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
