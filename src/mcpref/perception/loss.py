"""(10)-(12) of specs/perception.md: the P1 loss of each clip, METRA's perception loss
without the skill, with the dual variable log(lambda)."""

import math

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

    def clip_loss(self, z: Tensor, clip: Clip) -> Tensor:
        """(11) of one clip, z of shape (L, D), in the dtype of z."""
        cfg, like = self.cfg, {"dtype": z.dtype, "device": z.device}
        frames, metas = clip_states(clip, cfg, self.sprite)
        chi = torch.tensor(changed(frames, metas, cfg.tau), **like)
        n = torch.tensor(clip.info["n"], **like)
        alpha = torch.tensor(cfg.alpha, **like)
        d = torch.linalg.vector_norm(alpha * (z[1:] - z[:-1]), ord=2, dim=-1)
        q = torch.clamp(n.square() - d.square(), max=cfg.dual_slack)
        log_lambda = self.log_lambda.to(z.dtype)
        lam = log_lambda.exp().detach()
        pushed = chi * (-d - lam * q + log_lambda * q.detach())
        h = hinge(z, cfg.c)
        return (pushed + cfg.beta * (h[:-1] + h[1:])).mean()

    def forward(self, Z: Tensor, clips: list[Clip]) -> Tensor:
        """(11) of each clip, shape (n,)."""
        return torch.stack([self.clip_loss(z, clip) for z, clip in zip(Z, clips)])

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): phi_dual = [log(lambda)]."""
        return {"phi_dual": [self.log_lambda]}
