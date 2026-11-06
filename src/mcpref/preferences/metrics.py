"""Scalar preference metrics of specs/pretrain-diagnostics.md (5)-(8)."""

import math
from collections.abc import Iterable, Sequence


def mean_nullable(values: Iterable[float | None]) -> tuple[float | None, int]:
    """Mean and count of non-null values, or `(None, 0)` when empty."""
    found = [value for value in values if value is not None]
    return (sum(found) / len(found), len(found)) if found else (None, 0)


def quantile(values: list[float], q: float) -> float | None:
    """(6): the specified one-based linear interpolation, without a library rule."""
    if not values:
        return None
    ordered = sorted(values)
    h = 1 + q * (len(ordered) - 1)
    j = math.floor(h)
    alpha = h - j
    if alpha == 0:
        return ordered[j - 1]
    return (1 - alpha) * ordered[j - 1] + alpha * ordered[j]


def calibration_bins(
    count: Sequence[int],
    prediction: Sequence[float],
    target: Sequence[float],
    bins: int,
) -> tuple[list[dict], float | None]:
    """(7): fixed-width calibration from sufficient bin sums."""
    total = sum(count)
    rows, ece = [], 0.0
    for b in range(bins):
        n = count[b]
        mean_prediction = prediction[b] / n if n else None
        mean_target = target[b] / n if n else None
        rows.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "count": n,
                "mean_prediction": mean_prediction,
                "mean_target": mean_target,
            }
        )
        if n:
            ece += n / total * abs(mean_prediction - mean_target)
    return rows, ece if total else None


def calibration(
    predictions: list[float], targets: list[float], bins: int
) -> tuple[list[dict], float | None]:
    """(7): calibration of equally weighted predictions and targets."""
    count = [0] * bins
    prediction_sum, target_sum = [0.0] * bins, [0.0] * bins
    for prediction, target in zip(predictions, targets):
        b = min(int(prediction * bins), bins - 1)
        count[b] += 1
        prediction_sum[b] += prediction
        target_sum[b] += target
    return calibration_bins(count, prediction_sum, target_sum, bins)


def head_suite(
    predictions: list[float],
    targets: list[float],
    *,
    full: bool,
    calibration_bins_count: int,
    saturation_threshold: float,
    quantiles: tuple[float, ...],
) -> dict:
    """(5): a step or full preference-head output suite."""
    count = len(predictions)
    mean = sum(predictions) / count if count else None
    std = (
        math.sqrt(sum((value - mean) ** 2 for value in predictions) / count)
        if count
        else None
    )
    result = {
        "count": count,
        "mean": mean,
        "std": std,
        "brier": (
            sum((u - y) ** 2 for u, y in zip(predictions, targets)) / count
            if count
            else None
        ),
        "low": (
            sum(u < saturation_threshold for u in predictions) / count
            if count
            else None
        ),
        "high": (
            sum(u > 1 - saturation_threshold for u in predictions) / count
            if count
            else None
        ),
    }
    if full:
        bins, ece = calibration(predictions, targets, calibration_bins_count)
        result |= {
            "quantiles": {str(q): quantile(predictions, q) for q in quantiles},
            "calibration": bins,
            "ece": ece,
        }
    return result


def inc_classification(
    predictions: list[float], targets: list[float], threshold: float
) -> dict:
    """(8): incomparability confusion counts, recalls and accuracies."""
    tp = sum(u >= threshold and y == 1 for u, y in zip(predictions, targets))
    tn = sum(u < threshold and y == 0 for u, y in zip(predictions, targets))
    fp = sum(u >= threshold and y == 0 for u, y in zip(predictions, targets))
    fn = sum(u < threshold and y == 1 for u, y in zip(predictions, targets))
    count = len(predictions)
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / count if count else None,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": (
            (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
        ),
    }
