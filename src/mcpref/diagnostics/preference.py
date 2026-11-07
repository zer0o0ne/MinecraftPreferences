"""Producer-owned preference records of specs/rl-diagnostics.md (6)-(7)."""

import json
from collections.abc import Sequence

import torch

from mcpref import REPO
from mcpref.diagnostics.config import DiagnosticsConfig
from mcpref.diagnostics.stats import distribution, rate
from mcpref.diagnostics.types import (
    MetricRecord,
    ProducerContext,
    ReferencePair,
    metric_record,
)
from mcpref.preferences.diagnostics import GlobalFit, PreferenceRows
from mcpref.preferences.loss import bce, preference_baselines


def preference_values(rows: PreferenceRows) -> dict[str, list[float]]:
    inc = rows.inc.to(rows.t.dtype)
    p = rows.p.to(rows.s.dtype)
    inc_loss = 2 * bce(rows.t, inc)
    succ_loss = 2 * torch.where(rows.inc == 0, bce(rows.s, p), 0.0)
    return {
        "P": (inc_loss + succ_loss).tolist(),
        "P_inc": inc_loss.tolist(),
        "P_succ": succ_loss.tolist(),
    }


def preference_record(
    rows: PreferenceRows | None, cfg: DiagnosticsConfig, context: ProducerContext
) -> MetricRecord:
    reason = None if rows is not None else "empty_pair_population"
    values = (
        preference_values(rows)
        if rows is not None
        else {"P": [], "P_inc": [], "P_succ": []}
    )
    metrics = {
        key: distribution(value, cfg, reason=reason) for key, value in values.items()
    }
    populations = {key: list(context.cluster_ids) for key in values}
    return metric_record(
        "pretrain", f"step:{context.steps[0]}", context, metrics, populations
    )


def add_model_metrics(
    record: MetricRecord,
    rows: PreferenceRows | None,
    cfg: DiagnosticsConfig,
    context: ProducerContext,
) -> None:
    comparable = (
        rows.inc == 0 if rows is not None else torch.tensor([], dtype=torch.bool)
    )
    probabilities = (
        torch.sigmoid(rows.s[comparable]).tolist() if rows is not None else []
    )
    clusters = [
        cluster
        for cluster, is_comparable in zip(context.cluster_ids, comparable.tolist())
        if is_comparable
    ]
    for name, threshold, above in (
        ("target.model_lt_0_2", 0.2, False),
        ("target.model_gt_0_8", 0.8, True),
    ):
        values = [
            float(value > threshold) if above else float(value < threshold)
            for value in probabilities
        ]
        record["metrics"][name] = rate(values, clusters, cfg)
        record["populations"][name] = list(clusters)


def on_pretrain_step(
    rows: PreferenceRows | None,
    *,
    fit: GlobalFit,
    cfg: DiagnosticsConfig,
    context: ProducerContext,
) -> MetricRecord:
    """(7): corrected joint-P progress against fixed global-train constants."""
    record = preference_record(rows, cfg, context)
    model_loss = record["metrics"]["P"]["mean"]
    baselines = preference_baselines(rows, fit)
    baseline = baselines["inc_only"]
    floor = (
        baselines["inc_constant"] + baselines["target_entropy"]
        if baselines["inc_constant"] is not None
        and baselines["target_entropy"] is not None
        else None
    )
    reason = None
    if rows is None:
        reason = "empty_pair_population"
    elif baseline is None or floor is None:
        reason = "undefined_baseline"
    elif baseline == floor:
        reason = "baseline_equals_floor"
    progress = (
        (baseline - model_loss) / (baseline - floor)
        if reason is None and model_loss is not None
        else None
    )
    record["metrics"]["pretrain.progress"] = distribution(
        [] if progress is None else [progress],
        cfg,
        missing=int(progress is None),
        reason=reason,
    )
    record["populations"]["pretrain.progress"] = list(context.cluster_ids)
    return record


def delta_distribution(
    pairs: Sequence[ReferencePair] | None,
    field: str,
    applicable: bool,
    missing: int,
    cfg: DiagnosticsConfig,
) -> dict:
    if not applicable:
        return distribution([], cfg, missing=missing, reason="not_applicable_condition")
    if pairs is None:
        return distribution(
            [], cfg, missing=missing, reason="legacy_reference_pairs_absent"
        )
    return distribution([getattr(pair, field) for pair in pairs], cfg)


def on_reference(
    rows: PreferenceRows | None,
    *,
    pairs: Sequence[ReferencePair] | None,
    cfg: DiagnosticsConfig,
    context: ProducerContext,
) -> MetricRecord:
    """Saved reference rows and typed pairs under exact split/condition semantics."""
    record = preference_record(rows, cfg, context)
    shard = int(context.artifact_paths[0].split("/")[-2])
    record["id"] = f"reference:{shard}:{context.split}:{context.condition}"
    if context.split == "validation":
        add_model_metrics(record, rows, cfg, context)
    applicable = {
        "delta_only_phi": context.condition == "phi_only",
        "delta_joint": context.condition == "joint",
    }
    for field, applies in applicable.items():
        record["metrics"][field] = delta_distribution(
            pairs, field, applies, len(context.cluster_ids), cfg
        )
        record["populations"][field] = (
            list(context.cluster_ids) if applies and pairs is not None else []
        )
    return record


def unique(entries: list[dict], key: str, value: int) -> dict:
    found = [entry for entry in entries if entry.get(key) == value]
    if len(found) != 1:
        raise RuntimeError(
            f"expected exactly one {key}={value} entry, found {len(found)}"
        )
    return found[0]


def join_pretrain_step(record: MetricRecord, sources: Sequence[str]) -> MetricRecord:
    """Join exact complete outputs and only the uniquely closing window."""
    step = record["context"]["steps"][0]
    losses = json.loads((REPO / sources[0]).read_bytes())
    entry = unique(losses, "step", step)
    record["joined"]["outputs"] = entry["outputs"]
    if len(sources) == 1:
        record["joined"] |= {"window_preference": None, "reason": "window_open"}
        return record
    windows = json.loads((REPO / sources[1]).read_bytes())
    window = unique(windows, "last_step", step)
    record["joined"] |= {"window_preference": window["preference"], "reason": None}
    return record


def join_reference(record: MetricRecord, sources: Sequence[str]) -> MetricRecord:
    """Join exactly conditions[condition][split], including late-initial losses."""
    saved = json.loads((REPO / sources[0]).read_bytes())
    context = record["context"]
    record["joined"]["condition"] = saved["conditions"][context["condition"]][
        context["split"]
    ]
    return record
