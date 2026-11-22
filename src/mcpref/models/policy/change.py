"""Per-action change weighting with separate policy and change gradients."""

import torch
from torch import Tensor
from torch.nn import functional as F


def change_loss(probability: Tensor, repeated: Tensor, action_loss: Tensor) -> Tensor:
    """Value e*|L|; action-loss gradient e*dL, change gradient |L|*de."""
    error = torch.where(repeated, probability, 1 - probability)
    magnitude = action_loss + (action_loss.abs() - action_loss).detach()
    return error * magnitude


def change_cross_entropy(
    logits: Tensor, q: Tensor, probability: Tensor, repeated: Tensor, known: Tensor
) -> Tensor:
    """Weight each current occurrence by membership in the preceding interval."""
    losses = -F.log_softmax(logits, dim=-1)
    weighted = change_loss(probability[..., None], repeated, losses)
    return torch.where(known, (q * weighted).sum(dim=-1), 0.0)
