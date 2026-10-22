"""(10) of specs/preference.md: the loss of P over both orders of each pair."""

import torch
from torch import Tensor
from torch.nn import functional as F

PLACEHOLDER = 0.5
"""(10): the finite target of s_r for inc = 1, its term removed by torch.where."""


def bce(x: Tensor, y: Tensor) -> Tensor:
    """Notation BCE(x, y) on logits x and targets y, per entry."""
    return F.binary_cross_entropy_with_logits(
        x, y, weight=None, reduction="none", pos_weight=None
    )


class PreferenceLoss(torch.nn.Module):
    """(10): l^P_r = 2 BCE(t_r, inc) + 2 [inc = 0] BCE(s_r, p), shape (m,)."""

    def forward(self, out: tuple[Tensor, Tensor], labels: list[dict]) -> Tensor:
        s, t = out
        inc = t.new_tensor([y["inc"] for y in labels])
        p = s.new_tensor([y["p"] if y["inc"] == 0 else PLACEHOLDER for y in labels])
        succ = torch.where(inc == 0, bce(s, p), 0.0)
        return 2 * bce(t, inc) + 2 * succ

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): l^P declares no group."""
        return {}
