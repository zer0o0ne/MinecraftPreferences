"""(1)-(5) of specs/gui-stack.md: standardised features, the one-hidden-layer MLP, its
class-weighted loss, full-batch Adam and the class probabilities."""

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mcpref.stacking.config import GuiStackConfig


def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Notation: mu, s over the rows of x (|S| x C), NaN left out; s = 0 when the values
    of a column are all equal."""
    equal = np.nanmax(x, axis=0) == np.nanmin(x, axis=0)
    s = np.nanstd(x, axis=0, ddof=0)
    return np.nanmean(x, axis=0), np.where(equal, 0.0, s)


def standardised(x: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Notation: z = (x - mu) / s where x is not NaN and s > 0, else 0."""
    keep = ~np.isnan(x) & (s > 0)
    return np.divide(x - mu, s, out=np.zeros_like(x), where=keep)


def uniform(shape: tuple[int, ...], fan_in: int, gen: torch.Generator) -> torch.Tensor:
    """(2): (2u - 1) / sqrt(fan_in), u = torch.rand of the shape (float32)."""
    u = torch.rand(shape, generator=gen, dtype=torch.float32)
    return (2 * u - 1) / math.sqrt(fan_in)


def init_params(
    c: int, h: int, n_labels: int, seed: int, device: torch.device
) -> list[torch.Tensor]:
    """(2): [W1, b1, W2, b2] drawn in this order on the CPU, then moved to `device`."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [((h, c), c), ((h,), c), ((n_labels, h), h), ((n_labels,), h)]
    drawn = [uniform(shape, fan_in, gen) for shape, fan_in in shapes]
    return [p.to(device).requires_grad_() for p in drawn]


def logits(params: Sequence[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    """(1): l = W2 max(0, W1 z + b1) + b2, frames x |Y|."""
    w1, b1, w2, b2 = params
    return torch.relu(z @ w1.T + b1) @ w2.T + b2


def loss(
    params: Sequence[torch.Tensor],
    z: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    l2: float,
) -> torch.Tensor:
    """(3): sum_f w_f (-log p(y_f | z_f)) / sum_f w_f + l2 (||W1||_F^2 + ||W2||_F^2)."""
    log_p = F.log_softmax(logits(params, z), dim=1)
    nll = -log_p.gather(1, y.unsqueeze(1)).squeeze(1)
    w1, _, w2, _ = params
    return (w * nll).sum() / w.sum() + l2 * (w1.square().sum() + w2.square().sum())


def fit(
    params: Sequence[torch.Tensor],
    z: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    cfg: GuiStackConfig,
    step: Callable[[], None],
) -> tuple[float, float]:
    """(4): T full-batch Adam updates of (3) in place; returns L(theta_0), L(theta_T);
    step() after each update."""
    adam = torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.adam_eps,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        decoupled_weight_decay=False,
    )
    initial = loss(params, z, y, w, cfg.l2).item()
    for _ in range(cfg.steps):
        adam.zero_grad()
        loss(params, z, y, w, cfg.l2).backward()
        adam.step()
        step()
    return initial, loss(params, z, y, w, cfg.l2).item()


def probabilities(params: Sequence[torch.Tensor], z: torch.Tensor) -> np.ndarray:
    """(5): p_f, the softmax of the logits computed in float64, frames x |Y|."""
    ell = logits(params, z).detach().to(torch.float64)
    return torch.softmax(ell, dim=1).cpu().numpy()
