"""Preference observations, global-fit types and suites of pretrain diagnostics."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from mcpref.preferences.metrics import head_suite, inc_classification

BASELINE_KEYS = (
    "zero",
    "inc_only",
    "both",
    "target_entropy",
    "inc_constant",
    "succ_constant",
)


@dataclass(frozen=True)
class PreferenceRows:
    s: Tensor
    t: Tensor
    p: Tensor
    inc: Tensor


@dataclass(frozen=True)
class Stratum:
    key: str
    kind: Literal["comparable", "inc"]
    p: float | None
    train: int
    validation: int


@dataclass(frozen=True)
class GlobalFit:
    pairs: int
    incomparable: int
    comparable: int
    sum_p: float | None
    inc_probability: float
    succ_probability: float | None
    inc_logit: float | None
    succ_logit: float | None
    strata: tuple[Stratum, ...]


def pool_preferences(rows: Iterable[PreferenceRows | None]) -> PreferenceRows | None:
    """Concatenate non-null per-step preference rows in population order."""
    found = [row for row in rows if row is not None]
    if not found:
        return None
    return PreferenceRows(
        *(
            torch.cat([getattr(row, key) for row in found])
            for key in ("s", "t", "p", "inc")
        )
    )


def stratum_suite(
    rows: PreferenceRows | None, strata: tuple[Stratum, ...]
) -> list[dict]:
    """(9): every catalogue stratum, including empty populations."""
    result = []
    for stratum in strata:
        selected: Tensor | None = None
        target: Tensor | None = None
        if rows is not None:
            if stratum.kind == "inc":
                mask = rows.inc == 1
                selected, target = rows.t[mask], torch.ones_like(rows.t[mask])
            else:
                p = torch.tensor(stratum.p, dtype=torch.float32).item()
                mask = (rows.inc == 0) & (rows.p == p)
                selected, target = rows.s[mask], rows.p[mask]
        count = 0 if selected is None else selected.numel()
        prediction = torch.sigmoid(selected) if count else None
        result.append(
            {
                "key": stratum.key,
                "kind": stratum.kind,
                "p": stratum.p,
                "count": count,
                "mean_prediction": prediction.mean().item() if count else None,
                "mean_target": target.mean().item() if count else None,
                "bce": (
                    2
                    * torch.nn.functional.binary_cross_entropy_with_logits(
                        selected,
                        target,
                        weight=None,
                        pos_weight=None,
                        reduction="none",
                    )
                    .mean()
                    .item()
                    if count
                    else None
                ),
                "brier": (prediction - target).square().mean().item()
                if count
                else None,
            }
        )
    return result


def preference_suite(
    rows: PreferenceRows | None,
    *,
    strata: tuple[Stratum, ...],
    calibration_bins: int,
    saturation_threshold: float,
    inc_classification_threshold: float,
    quantiles: tuple[float, ...],
    full: bool,
) -> dict:
    """(5)-(9): step or full preference output suites."""
    if rows is None:
        inc_predictions: list[float] = []
        inc_targets: list[float] = []
        succ_predictions: list[float] = []
        succ_targets: list[float] = []
    else:
        comparable = rows.inc == 0
        inc_predictions = torch.sigmoid(rows.t).tolist()
        inc_targets = rows.inc.to(torch.float32).tolist()
        succ_predictions = torch.sigmoid(rows.s[comparable]).tolist()
        succ_targets = rows.p[comparable].tolist()
    options = {
        "full": full,
        "calibration_bins_count": calibration_bins,
        "saturation_threshold": saturation_threshold,
        "quantiles": quantiles,
    }
    inc = head_suite(inc_predictions, inc_targets, **options)
    succ = (
        head_suite(succ_predictions, succ_targets, **options)
        if succ_predictions
        else None
    )
    if not full:
        return {"inc": inc if rows is not None else None, "succ": succ}
    return {
        "inc": inc,
        "succ": succ,
        "inc_classification": inc_classification(
            inc_predictions, inc_targets, inc_classification_threshold
        ),
        "strata": stratum_suite(rows, strata),
    }
