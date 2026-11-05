"""(3), (4) of specs/ppo.md: per-action PPO and factual-shift losses."""

from collections.abc import Callable

import torch
from torch import Tensor


def ppo_loss(
    logits: Tensor,
    actions: Tensor,
    behavior: Tensor,
    advantages: Tensor,
    clip: float,
) -> Tensor:
    """(3): the clipped policy loss of each action, shape (m,)."""
    rows = torch.arange(0, logits.shape[0], 1, dtype=torch.int64, device=logits.device)
    log_probability = torch.log_softmax(logits, dim=-1, dtype=None)[rows, actions]
    ratio = torch.exp(log_probability) / behavior.detach()
    advantage = advantages.detach()
    clipped = torch.clamp(ratio, min=1 - clip, max=1 + clip)
    return -torch.minimum(ratio * advantage, clipped * advantage)


def shift_loss(
    predicted: Tensor, actual: Tensor, metric: Callable[[Tensor], Tensor]
) -> Tensor:
    """(4): the squared metric from the detached factual shift, shape (m,)."""
    return metric(predicted - actual.detach()).square()
