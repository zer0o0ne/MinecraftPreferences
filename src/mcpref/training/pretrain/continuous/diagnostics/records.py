"""Shared diagnostic records with explicit continuous event populations."""

import json

import torch

from mcpref.config import REPO
from mcpref.diagnostics.core.types import ProducerContext
from mcpref.diagnostics.preference.baselines import preference_baselines
from mcpref.diagnostics.preference.suites import (
    pool_preferences,
    preference_suite,
)
from mcpref.diagnostics.pretrain.observation import preference_observation
from mcpref.diagnostics.producers.preference import (
    on_pretrain_step,
    preference_record,
    unique,
)
from mcpref.training.pretrain.continuous.diagnostics.observations import OwnSummary
from mcpref.training.pretrain.continuous.diagnostics.policy import add_policy


def preference_rows(result):
    if not result.preference_rows:
        return None
    rows = result.preference_rows
    return preference_observation(
        (
            torch.stack([r["s"] for r in rows]),
            torch.stack([r["t"] for r in rows]),
            [r["edge"]["labels"] for r in rows],
        )
    )


def preference_metrics(cfg, fit, rows, full):
    return preference_suite(
        rows,
        strata=fit.strata if fit else (),
        calibration_bins=cfg.calibration_bins,
        saturation_threshold=cfg.saturation_threshold,
        inc_classification_threshold=cfg.inc_classification_threshold,
        quantiles=cfg.quantiles,
        full=full,
    )


def decorate(context, result, fit):
    own = OwnSummary(context.cfg)
    own.add(result)
    metrics = own.finish(context.modules)
    rows = preference_rows(result)
    suite = preference_metrics(context.cfg, fit, rows, False)
    record = result.record(context.step)
    record.update({key: metrics["losses"][key] for key in ("phi", "pi", "P")})
    record.update(
        {
            "outputs": {
                "P_inc": suite["inc"],
                "P_succ": suite["succ"],
                "pi_action": metrics["pi"]["action"],
            },
            "phi_stats": metrics["phi"],
            "baselines": preference_baselines(rows, fit) if fit else None,
        }
    )
    return record, rows


def window_metrics(context, entries, fit):
    summary = OwnSummary(context.cfg)
    for _, result in entries:
        summary.add(result)
    rows = pool_preferences(preference_rows(result) for _, result in entries)
    return {
        "first_step": entries[0][0],
        "last_step": entries[-1][0],
        **summary.finish(context.modules),
        "preference": preference_metrics(context.cfg, fit, rows, True),
        "mean_weighted_loss": sum(
            result.record(step)["loss"] for step, result in entries
        )
        / len(entries),
    }


def endpoint_source(index, key):
    body = index.db.execute("SELECT body FROM endpoints WHERE id=?", (key,)).fetchone()[
        0
    ]
    return json.loads(body)["source"]


def stage_step(diagnostics, context, fit, result, rows, sources, channel):
    clusters = tuple(
        endpoint_source(context.index, row["edge"]["a"])
        for row in result.preference_rows
    )
    producer = ProducerContext(
        "pretrain-continuous",
        "train",
        context.phase,
        "pretrain",
        str(context.root.relative_to(REPO)),
        {},
        (context.step,),
        clusters,
        tuple(sources),
    )
    record = (
        on_pretrain_step(rows, fit=fit, cfg=diagnostics.cfg, context=producer)
        if fit is not None
        else preference_record(rows, diagnostics.cfg, producer)
    )
    if (
        context.step % context.cfg.checkpoint_steps == 0
        or context.step == context.total
    ):
        add_policy(record, result.ce_rows, producer, context.cfg, diagnostics.cfg)
    record["id"] = f"continuous:{context.phase}:step:{context.step}"
    diagnostics.stage(
        channel, context.step, f"step:{context.step}", context.step + 1, sources, record
    )


def join_step(record, sources):
    step = record["context"]["steps"][0]
    entry = unique(json.loads((REPO / sources[0]).read_bytes()), "step", step)
    record["joined"].update(
        {
            "outputs": entry["outputs"],
            "counts": entry["counts"],
            "window_preference": None,
            "reason": "window_open",
        }
    )
    if len(sources) > 1:
        window = unique(json.loads((REPO / sources[1]).read_bytes()), "last_step", step)
        record["joined"].update(window_preference=window["preference"], reason=None)
    return record
