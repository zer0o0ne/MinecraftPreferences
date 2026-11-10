"""(10)-(12) of specs/perception.md: the P1 loss of each clip, METRA's perception loss
without the skill, with the dual variable log(lambda)."""

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from mcpref.datasets.clips.io import Clip
from mcpref.models.perception.config import PerceptionConfig
from mcpref.models.perception.cursor import load_sprite
from mcpref.models.perception.meta import clip_states


def changed(frames: list[np.ndarray], metas: list[str], tau: float) -> list[bool]:
    """(10): chi_s, the meta changed or the mean absolute pixel difference > tau."""
    return [
        mu != mu_next or np.abs(v_next.astype(np.int64) - v).sum() / v.size > tau
        for v, v_next, mu, mu_next in zip(frames, frames[1:], metas, metas[1:])
    ]


def latent_metric(cfg: PerceptionConfig) -> Callable[[Tensor], Tensor]:
    """Interface: delta -> d_alpha = ||alpha * delta||_2 over the last axis."""
    cache = {}

    def d_alpha(delta: Tensor) -> Tensor:
        key = (delta.device, delta.dtype)
        alpha = cache.get(key)
        if alpha is None:
            alpha = torch.tensor(cfg.alpha, dtype=delta.dtype, device=delta.device)
            cache[key] = alpha
        return torch.linalg.vector_norm(alpha * delta, ord=2, dim=-1)

    return d_alpha


def hinge(z: Tensor, c: float) -> Tensor:
    """(11): max(0, ||z_s||^2 - c^2) of each state."""
    return torch.clamp(z.square().sum(dim=-1) - c**2, min=0)


def _terms(
    z: Tensor,
    n: Tensor,
    changed_rows: Tensor,
    log_lambda: Tensor,
    metric: Callable[[Tensor], Tensor],
    c: float,
    beta: float,
    dual_slack: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    chi = changed_rows.to(z.dtype)
    d = metric(z[1:] - z[:-1])
    q = torch.clamp(n.square() - d.square(), max=dual_slack)
    log_lambda = log_lambda.to(z.dtype)
    lam = log_lambda.exp().detach()
    push = -chi * d
    constraint = -chi * lam * q
    dual = chi * log_lambda * q.detach()
    pushed = chi * (-d - lam * q + log_lambda * q.detach())
    h = hinge(z, c)
    norm = beta * (h[:-1] + h[1:])
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


_COMPILED_TERMS = torch.compile(
    _terms,
    backend="inductor",
    fullgraph=True,
    dynamic=False,
    options={"compile_threads": 1},
)


class PhiLoss(torch.nn.Module):
    """(11): l^phi of each clip; log(lambda) starts at ln(lambda_0)."""

    def __init__(self, cfg: PerceptionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.sprite = load_sprite(cfg)
        self.metric = latent_metric(cfg)
        self.log_lambda = torch.nn.Parameter(torch.tensor(math.log(cfg.dual_init)))

    def _clip_terms(
        self,
        z: Tensor,
        clip: Clip,
        states: tuple[list[np.ndarray], list[str]] | None,
        core: Callable,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        cfg, like = self.cfg, {"dtype": z.dtype, "device": z.device}
        states = clip_states(clip, cfg, self.sprite) if states is None else states
        frames, metas = states
        changed_rows = torch.tensor(changed(frames, metas, cfg.tau), device=z.device)
        n = torch.tensor(clip.info["n"], **like)
        return core(
            z,
            n,
            changed_rows,
            self.log_lambda,
            self.metric,
            cfg.c,
            cfg.beta,
            cfg.dual_slack,
        )

    def clip_terms(
        self,
        z: Tensor,
        clip: Clip,
        states: tuple[list[np.ndarray], list[str]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """(4) of specs/pretrain-diagnostics.md: one clip's terms and observations."""
        return self._clip_terms(z, clip, states, _terms)

    def compiled_clip_terms(
        self,
        z: Tensor,
        clip: Clip,
        states: tuple[list[np.ndarray], list[str]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The same tensor terms, compiled for ordinary CUDA pretraining."""
        return self._clip_terms(z, clip, states, _COMPILED_TERMS)

    def _terms(
        self,
        Z: Tensor,
        clips: list[Clip],
        states: list[tuple[list[np.ndarray], list[str]]] | None,
        core: Callable,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        states = (
            [clip_states(clip, self.cfg, self.sprite) for clip in clips]
            if states is None
            else states
        )
        results = [
            self._clip_terms(z, clip, state, core)
            for z, clip, state in zip(Z, clips, states)
        ]
        terms = {
            key: torch.stack([result[0][key] for result in results])
            for key in results[0][0]
        }
        diagnostics = {
            key: torch.stack([result[1][key] for result in results])
            for key in results[0][1]
        }
        return terms, diagnostics

    def terms(
        self,
        Z: Tensor,
        clips: list[Clip],
        states: list[tuple[list[np.ndarray], list[str]]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self._terms(Z, clips, states, _terms)

    def compiled_terms(
        self,
        Z: Tensor,
        clips: list[Clip],
        states: list[tuple[list[np.ndarray], list[str]]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The same tensor terms, compiled for ordinary CUDA pretraining."""
        return self._terms(Z, clips, states, _COMPILED_TERMS)

    def clip_loss(self, z: Tensor, clip: Clip) -> Tensor:
        return self.clip_terms(z, clip)[0]["total"]

    def forward(self, Z: Tensor, clips: list[Clip]) -> Tensor:
        """(11) of each clip, shape (n,)."""
        return self.terms(Z, clips)[0]["total"]

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): phi_dual = [log(lambda)]."""
        return {"phi_dual": [self.log_lambda]}
