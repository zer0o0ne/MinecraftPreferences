"""Fixed preference baselines of pretrain-diagnostics.md."""

import math

import torch

from mcpref.diagnostics.preference.suites import (
    BASELINE_KEYS,
    GlobalFit,
    PreferenceRows,
)


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
