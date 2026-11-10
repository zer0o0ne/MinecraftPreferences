"""Preference loss of specs/preference.md."""

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


def _inputs(
    out: tuple[Tensor, Tensor], labels: list[dict]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    s, t = out
    inc = t.new_tensor([y["inc"] for y in labels])
    p = s.new_tensor([y["p"] if y["inc"] == 0 else PLACEHOLDER for y in labels])
    return s, t, inc, p


def _terms(s: Tensor, t: Tensor, inc: Tensor, p: Tensor) -> dict[str, Tensor]:
    succ = torch.where(inc == 0, bce(s, p), 0.0)
    inc_loss, succ_loss = 2 * bce(t, inc), 2 * succ
    return {"inc": inc_loss, "succ": succ_loss, "total": inc_loss + succ_loss}


_COMPILED_TERMS = torch.compile(
    _terms,
    backend="inductor",
    fullgraph=True,
    dynamic=False,
    options={"compile_threads": 1},
)


class PreferenceLoss(torch.nn.Module):
    """(10): l^P_r = 2 BCE(t_r, inc) + 2 [inc = 0] BCE(s_r, p), shape (m,)."""

    def terms(
        self, out: tuple[Tensor, Tensor], labels: list[dict]
    ) -> dict[str, Tensor]:
        """(2) of specs/pretrain-diagnostics.md: per-pair loss terms."""
        return _terms(*_inputs(out, labels))

    def compiled_terms(
        self, out: tuple[Tensor, Tensor], labels: list[dict]
    ) -> dict[str, Tensor]:
        """The same tensor terms, compiled for ordinary CUDA pretraining."""
        return _COMPILED_TERMS(*_inputs(out, labels))

    def forward(self, out: tuple[Tensor, Tensor], labels: list[dict]) -> Tensor:
        return self.terms(out, labels)["total"]

    def groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """(12): l^P declares no group."""
        return {}
