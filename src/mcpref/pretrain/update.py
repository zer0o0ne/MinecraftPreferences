"""(4)-(6) of specs/pretrain.md: the update of a step, gradient clipping, the learning
rate and AdamW over the groups."""

import math

import torch

from mcpref.pretrain.checks import Groups
from mcpref.pretrain.config import PretrainConfig


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
