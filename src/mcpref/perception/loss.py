"""(10)-(12) of specs/perception.md: the P1 loss of each clip, METRA's perception loss
without the skill, with the dual variable log(lambda)."""

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from mcpref.perception.config import PerceptionConfig
from mcpref.perception.cursor import load_sprite
from mcpref.perception.meta import clip_states
from mcpref.pretrain.clip import Clip


def changed(frames: list[np.ndarray], metas: list[str], tau: float) -> list[bool]:
    """(10): chi_s, the meta changed or the mean absolute pixel difference > tau."""
    return [
        mu != mu_next or np.abs(v_next.astype(np.int64) - v).sum() / v.size > tau
        for v, v_next, mu, mu_next in zip(frames, frames[1:], metas, metas[1:])
    ]


def latent_metric(cfg: PerceptionConfig) -> Callable[[Tensor], Tensor]:
    """Interface: delta -> d_alpha = ||alpha * delta||_2 over the last axis, alpha in
    delta's dtype on its device."""

    def d_alpha(delta: Tensor) -> Tensor:
        alpha = torch.tensor(cfg.alpha, dtype=delta.dtype, device=delta.device)
        return torch.linalg.vector_norm(alpha * delta, ord=2, dim=-1)

    return d_alpha


def hinge(z: Tensor, c: float) -> Tensor:
    """(11): max(0, ||z_s||^2 - c^2) of each state."""
    return torch.clamp(z.square().sum(dim=-1) - c**2, min=0)


class PhiLoss(torch.nn.Module):
    """(11): l^phi of each clip; log(lambda) starts at ln(lambda_0)."""

    def __init__(self, cfg: PerceptionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sprite = load_sprite(cfg)
        self.log_lambda = torch.nn.Parameter(torch.tensor(math.log(cfg.dual_init)))

    def clip_terms(
        self, z: Tensor, clip: Clip
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """(4) of specs/pretrain-diagnostics.md: one clip's terms and observations."""
        cfg, like = self.cfg, {"dtype": z.dtype, "device": z.device}
        frames, metas = clip_states(clip, cfg, self.sprite)
        changed_rows = torch.tensor(changed(frames, metas, cfg.tau), device=z.device)
        chi = changed_rows.to(z.dtype)
        n = torch.tensor(clip.info["n"], **like)
        d = latent_metric(cfg)(z[1:] - z[:-1])
        q = torch.clamp(n.square() - d.square(), max=cfg.dual_slack)
        log_lambda = self.log_lambda.to(z.dtype)
        lam = log_lambda.exp().detach()
        push = -chi * d
        constraint = -chi * lam * q
        dual = chi * log_lambda * q.detach()
        pushed = chi * (-d - lam * q + log_lambda * q.detach())
        h = hinge(z, cfg.c)
        norm = cfg.beta * (h[:-1] + h[1:])
        terms = {
            "push": push.mean(),
            "constraint": constraint.mean(),
            "dual": dual.mean(),
            "norm": norm.mean(),
            "total": (pushed + norm).mean(),
        }
        diagnostics = {
            "changed": changed_rows,
            "d_alpha": d,
            "z_norm": torch.linalg.vector_norm(z, ord=2, dim=-1),
            "hinge": h,
        }
        return terms, diagnostics

    def terms(
        self, Z: Tensor, clips: list[Clip]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        results = [self.clip_terms(z, clip) for z, clip in zip(Z, clips)]
        terms = {
            key: torch.stack([result[0][key] for result in results])
            for key in results[0][0]
        }
        diagnostics = {
            key: torch.stack([result[1][key] for result in results])
            for key in results[0][1]
        }
        return terms, diagnostics

    def clip_loss(self, z: Tensor, clip: Clip) -> Tensor:
        return self.clip_terms(z, clip)[0]["total"]

    def forward(self, Z: Tensor, clips: list[Clip]) -> Tensor:
        """(11) of each clip, shape (n,)."""
        return self.terms(Z, clips)[0]["total"]

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): phi_dual = [log(lambda)]."""
        return {"phi_dual": [self.log_lambda]}
