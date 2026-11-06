"""Accepted-step and pooled-pair summaries of specs/pretrain-diagnostics.md (28)."""

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from mcpref.preferences.metrics import quantile
from mcpref.pretrain.gradient_pairs import pair_distribution

if TYPE_CHECKING:
    from mcpref.pretrain.gradient_flow import GradientStepRecord

METRIC_PATHS = (
    "phi.total.norms.phi",
    "phi.total.norms.pi",
    "phi.total.norms.P",
    "phi.total.dot.P_phi",
    "phi.total.dot.P_pi",
    "phi.total.dot.P_other",
    "phi.total.dot.P_joint",
    "phi.total.cosine.P_phi",
    "phi.total.cosine.P_pi",
    "phi.total.cosine.P_other",
    "phi.total.update_dot",
    "P.total.norm",
    "P.total.update_dot",
    "counterfactual.delta_only_phi.mean",
    "counterfactual.delta_joint.mean",
)


def at(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return value


def step_distribution(values: list[float], quantiles: tuple[float, ...]) -> dict:
    count = len(values)
    return {
        "count": count,
        "median": quantile(values, 0.5),
        "quantiles": {str(q): quantile(values, q) for q in quantiles},
        "fraction_negative": sum(value < 0 for value in values) / count
        if count
        else None,
        "fraction_positive": sum(value > 0 for value in values) / count
        if count
        else None,
        "fraction_zero": sum(value == 0 for value in values) / count if count else None,
    }


def summarize(
    records: list["GradientStepRecord"], quantiles: tuple[float, ...] | None = None
) -> dict:
    """(28): regenerate summary solely from accepted real checkpoint records."""
    records = [
        record
        for record in records
        if record.metrics["evidence"]["kind"] == "real_pretrain_step"
    ]
    if quantiles is None:
        quantiles = (
            tuple(
                float(key)
                for key in records[0].metrics["counterfactual"]["delta_only_phi"][
                    "quantiles"
                ]
            )
            if records
            else ()
        )
    rows = []
    for record in records:
        metrics = record.metrics
        rows.append(
            {
                "step": metrics["step"],
                "roles": metrics["roles"],
                "metrics": str(
                    PurePosixPath(metrics["evidence"]["pairs"]).with_name(
                        "metrics.json"
                    )
                ),
                "phi_total": metrics["phi"]["total"],
                "P_total": metrics["P"]["total"],
                "delta_only_phi_mean": metrics["counterfactual"]["delta_only_phi"][
                    "mean"
                ],
                "delta_joint_mean": metrics["counterfactual"]["delta_joint"]["mean"],
            }
        )
    distributions = {}
    for path in METRIC_PATHS:
        values = [at(record.metrics, path) for record in records]
        values = [value for value in values if value is not None]
        distributions[path] = step_distribution(values, quantiles)
    pairs = {
        key: pair_distribution(
            [pair[key] for record in records for pair in record.pairs], quantiles
        )
        for key in ("delta_only_phi", "delta_joint")
    }
    return {
        "version": 1,
        "steps": rows,
        "step_distributions": distributions,
        "pair_distributions": pairs,
    }
