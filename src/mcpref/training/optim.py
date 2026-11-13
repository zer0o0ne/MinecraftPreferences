"""AdamW construction, learning-rate updates and update reconstruction."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from mcpref.infrastructure.profiling.scopes import measured, phase
from mcpref.training.groups import Groups
from mcpref.training.pretrain.config import PretrainConfig


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
    fused = torch.device(cfg.device).type == "cuda"
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=cfg.betas,
        eps=cfg.adam_eps,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=fused,
    )

    def restore_backend(_, state: dict) -> dict:
        return state | {
            "param_groups": [
                group | {"foreach": False, "fused": fused}
                for group in state["param_groups"]
            ]
        }

    optimizer.register_load_state_dict_pre_hook(restore_backend)
    return optimizer


@measured("optimizer")
def update(
    cfg: PretrainConfig,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    t: int,
    total: int,
) -> dict:
    """(4)-(6) of step t: clipping over G, eta_{q,t} set, AdamW step; {q: eta_{q,t}}."""
    params = [p for g in optimizer.param_groups for p in g["params"]]
    with phase("optimizer.clip_grad"):
        torch.nn.utils.clip_grad_norm_(
            params, max_norm=cfg.grad_clip, norm_type=2.0, error_if_nonfinite=False
        )
    f, lr = lr_factor(t, total, cfg.warmup_steps, cfg.lr_floor), {}
    for g, q in zip(optimizer.param_groups, groups):
        g["lr"] = lr[q] = cfg.groups[q]["lr"] * f
    with phase("optimizer.adamw"):
        optimizer.step()
    return lr


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
