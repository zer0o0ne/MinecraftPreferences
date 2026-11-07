"""Online distributions, Wilson intervals and real-valued cluster bootstrap."""

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

from mcpref.diagnostics.config import DiagnosticsConfig
from mcpref.preferences.metrics import quantile


def wilson(n: int, numerator: int, level: float) -> dict:
    if n == 0:
        return {"low": None, "high": None}
    z = NormalDist().inv_cdf((1 + level) / 2)
    p = numerator / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    delta = z / denominator * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return {"low": max(0, center - delta), "high": min(1, center + delta)}


def cluster_bootstrap(
    values: Sequence[float], cluster_ids: Sequence[str], cfg: DiagnosticsConfig
) -> dict:
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must be aligned")
    clusters = list(dict.fromkeys(cluster_ids))
    count = len(clusters)
    result = {"low": None, "high": None}
    if count < cfg.min_cluster_count or not values:
        return result
    positions = {cluster: [] for cluster in clusters}
    for position, cluster in enumerate(cluster_ids):
        positions[cluster].append(position)
    sums = np.array(
        [sum(values[position] for position in positions[c]) for c in clusters],
        dtype=np.float64,
    )
    sizes = np.array([len(positions[c]) for c in clusters], dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64([cfg.seed, 5]))
    replicates = []
    for _ in range(cfg.bootstrap_samples):
        chosen = generator.choice(
            count, size=count, replace=True, p=None, axis=0, shuffle=True
        )
        replicates.append(float(sums[chosen].sum() / sizes[chosen].sum()))
    result["low"] = quantile(replicates, (1 - cfg.ci_level) / 2)
    result["high"] = quantile(replicates, (1 + cfg.ci_level) / 2)
    return result


def rate(
    values: Sequence[float], cluster_ids: Sequence[str], cfg: DiagnosticsConfig
) -> dict:
    n = len(values)
    numerator = sum(values)
    if any(value not in (0, 1) for value in values):
        raise ValueError("rate observations must be binary")
    numerator = int(numerator)
    return {
        "n": n,
        "numerator": numerator,
        "value": numerator / n if n else None,
        "wilson": wilson(n, numerator, cfg.ci_level),
        "bootstrap": cluster_bootstrap(values, cluster_ids, cfg),
        "insufficient_clusters": len(set(cluster_ids)) < cfg.min_cluster_count,
    }


def distribution(
    values: Sequence[float],
    cfg: DiagnosticsConfig,
    *,
    missing: int = 0,
    reason: str | None = None,
) -> dict:
    materialized = list(values)
    n = len(materialized)
    return {
        "n": n,
        "mean": sum(materialized) / n if n else None,
        "median": quantile(materialized, 0.5),
        "quantiles": {str(q): quantile(materialized, q) for q in cfg.quantiles},
        "positive": sum(value > 0 for value in materialized) / n if n else None,
        "negative": sum(value < 0 for value in materialized) / n if n else None,
        "zero": sum(value == 0 for value in materialized) / n if n else None,
        "missing": missing,
        "reason": reason,
    }
