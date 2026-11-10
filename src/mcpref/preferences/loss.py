"""Preference loss of specs/preference.md and baselines of pretrain-diagnostics.md."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from mcpref.preferences.diagnostics import (
    BASELINE_KEYS,
    GlobalFit,
    PreferenceRows,
)

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


def probability_ce(target: float, probability: float) -> float | None:
    """Cross-entropy of one target, null when its exact value is infinite."""
    result = 0.0
    if target:
        if probability == 0:
            return None
        result -= target * math.log(probability)
    if 1 - target:
        if probability == 1:
            return None
        result -= (1 - target) * math.log(1 - probability)
    return result


def weighted_ce(targets: list[float], probability: float) -> float | None:
    """Mean cross-entropy at one fixed probability over an equally weighted list."""
    values = [probability_ce(target, probability) for target in targets]
    if any(value is None for value in values):
        return None
    return sum(values) / len(targets) if targets else 0.0


def preference_baselines(
    rows: PreferenceRows | None, fit: GlobalFit
) -> dict[str, float | None]:
    """(11): fixed global constants evaluated on one pair population."""
    if rows is None:
        return {key: None for key in BASELINE_KEYS}
    count = rows.inc.numel()
    inc_targets = rows.inc.to(torch.float32).tolist()
    comparable = rows.inc == 0
    p_targets = rows.p[comparable].tolist()
    comparable_weight = len(p_targets) / count
    inc_constant = weighted_ce(inc_targets, fit.inc_probability)
    inc_constant = 2 * inc_constant if inc_constant is not None else None
    succ_constant = (
        weighted_ce(p_targets, fit.succ_probability)
        if fit.succ_probability is not None
        else None
    )
    succ_constant = (
        2 * succ_constant * comparable_weight if succ_constant is not None else None
    )
    zero = 2 * math.log(2) * (1 + comparable_weight)
    inc_only = (
        inc_constant + 2 * math.log(2) * comparable_weight
        if inc_constant is not None
        else None
    )
    both = (
        inc_constant + succ_constant
        if inc_constant is not None and succ_constant is not None
        else None
    )
    target_entropy = 2 * sum(probability_ce(p, p) for p in p_targets) / count
    return {
        "zero": zero,
        "inc_only": inc_only,
        "both": both,
        "target_entropy": target_entropy,
        "inc_constant": inc_constant,
        "succ_constant": succ_constant,
    }
