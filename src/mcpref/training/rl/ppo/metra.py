"""(5), (6) of specs/ppo.md: the METRA skill and perception loss."""

import math

import torch
from torch import Tensor

from mcpref.training.rl.ppo.config import PpoConfig


def unit_skill(predicted: Tensor, alpha: Tensor, eps: float) -> Tensor:
    """(5): the detached prediction direction in the weighted geometry."""
    norm = torch.linalg.vector_norm(predicted * alpha, ord=2, dim=-1, keepdim=True)
    denominator = torch.clamp(norm, min=eps, max=None)
    return torch.divide(predicted.detach(), denominator.detach(), rounding_mode=None)


def metra_loss(
    actual: Tensor,
    skill: Tensor,
    alpha: Tensor,
    log_lambda: Tensor,
    slack: float,
) -> Tensor:
    """(6): the weighted skill inner product and METRA dual constraint."""
    weighted_actual = actual * alpha
    q = torch.clamp(
        1 - weighted_actual.square().sum(dim=-1, keepdim=False, dtype=None),
        min=None,
        max=slack,
    )
    dual = log_lambda.exp().detach()
    inner_product = (weighted_actual * (alpha * skill.detach())).sum(
        dim=-1, keepdim=False, dtype=None
    )
    return -inner_product - dual * q + log_lambda * q.detach()


class MetraLoss(torch.nn.Module):
    """(5), (6): METRA loss with its persistent trainable dual."""

    def __init__(self, cfg: PpoConfig):
        super().__init__()
        self.cfg = cfg
        self.log_lambda = torch.nn.Parameter(torch.tensor(math.log(cfg.dual_init)))

    def forward(self, actual: Tensor, predicted: Tensor, alpha: Tensor) -> Tensor:
        skill = unit_skill(predicted, alpha, self.cfg.skill_eps)
        return metra_loss(
            actual,
            skill,
            alpha,
            self.log_lambda.to(dtype=actual.dtype),
            self.cfg.dual_slack,
        )

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return {"metra_dual": [self.log_lambda]}
