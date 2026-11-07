"""Stage-33a outbox staging and exact accepted-artifact joins for pretraining."""

import json
from collections.abc import Sequence

from mcpref import REPO
from mcpref.diagnostics.preference import (
    join_pretrain_step,
    join_reference,
    on_pretrain_step,
    on_reference,
)
from mcpref.diagnostics.runtime import DiagnosticRun
from mcpref.diagnostics.types import ProducerContext, ReferencePair
from mcpref.preferences.diagnostics import GlobalFit, PreferenceRows
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.reference import ReferenceData
from mcpref.pretrain.reference_select import PARTS
from mcpref.shards.store import complete_shards, name

CONDITIONS = ("joint", "P_only", "phi_only", "initial")


def episode(path: str) -> str:
    return json.loads((REPO / path / "clip.json").read_bytes())["relpath"]


def step_sources(
    cfg: PretrainConfig, checkpoint: int, step: int, total: int
) -> list[str]:
    root = f"{cfg.out_dir}/checkpoints/{name(checkpoint)}"
    sources = [f"{root}/losses.json"]
    if step % cfg.diagnostic_window == 0 or step == total:
        sources.append(f"{root}/diagnostics.json")
    return sources


def stage_step(
    diagnostics: DiagnosticRun,
    cfg: PretrainConfig,
    fit: GlobalFit,
    checkpoint: int,
    step: int,
    total: int,
    batch: list[dict],
    rows: PreferenceRows | None,
) -> None:
    sources = step_sources(cfg, checkpoint, step, total)
    clusters = tuple(episode(item["a"]) for item in batch if "labels" in item)
    context = ProducerContext(
        "pretrain",
        "train",
        "joint",
        "pretrain",
        cfg.out_dir,
        {},
        (step,),
        clusters,
        tuple(sources),
    )
    record = on_pretrain_step(rows, fit=fit, cfg=diagnostics.cfg, context=context)
    diagnostics.stage("pretrain-step", step, f"step:{step}", step + 1, sources, record)


def accept_step(diagnostics: DiagnosticRun, step: int) -> None:
    diagnostics.accept("pretrain-step", f"step:{step}", join=join_pretrain_step)


def reference_sources(
    cfg: PretrainConfig, k: int, pairs: Sequence[ReferencePair] | None
) -> list[str]:
    root = f"{cfg.out_dir}/reference/{name(k)}"
    sources = [f"{root}/metrics.json"]
    if pairs is not None:
        sources.append(f"{root}/pairs.json")
    return sources


def stage_reference(
    diagnostics: DiagnosticRun,
    cfg: PretrainConfig,
    reference: ReferenceData,
    k: int,
    pairs: Sequence[ReferencePair] | None,
    observations: dict[str, dict[str, PreferenceRows | None]],
) -> None:
    sources = reference_sources(cfg, k, pairs)
    ordinal = k * 8
    for part in PARTS:
        indexed = [
            (index, item)
            for index, item in enumerate(reference.selection["items"])
            if item["part"] == part
        ]
        clusters = tuple(episode(item["a"]) for _, item in indexed)
        selected = None
        if pairs is not None:
            by_index = {pair.micro_batch: pair for pair in pairs}
            selected = tuple(by_index[index] for index, _ in indexed)
        for condition in CONDITIONS:
            context = ProducerContext(
                "pretrain-reference",
                part,
                condition,
                "pretrain",
                cfg.out_dir,
                {},
                (k * cfg.reference_steps,),
                clusters,
                tuple(sources),
            )
            record = on_reference(
                observations[condition][part],
                pairs=selected,
                cfg=diagnostics.cfg,
                context=context,
            )
            diagnostics.stage(
                "pretrain-reference",
                ordinal,
                f"reference:{k}",
                k + 1,
                sources,
                record,
            )
            ordinal += 1


def accept_reference(diagnostics: DiagnosticRun, k: int) -> None:
    diagnostics.accept("pretrain-reference", f"reference:{k}", join=join_reference)


def recover(diagnostics: DiagnosticRun) -> None:
    diagnostics.recover("pretrain-step", join=join_pretrain_step)
    diagnostics.recover("pretrain-reference", join=join_reference)


def register(
    cfg: PretrainConfig, diagnostics: DiagnosticRun, restored_step: int
) -> None:
    channels = diagnostics.state["channels"]
    step_watermark = channels.get("pretrain-step", {}).get(
        "watermark", restored_step + 1
    )
    diagnostics.register("pretrain-step", watermark=step_watermark, next=step_watermark)
    count = complete_shards(REPO / cfg.out_dir / "reference")
    reference_watermark = channels.get("pretrain-reference", {}).get("watermark", count)
    diagnostics.register(
        "pretrain-reference", watermark=reference_watermark, next=reference_watermark
    )
    recover(diagnostics)


def finish(total: int, cfg: PretrainConfig, diagnostics: DiagnosticRun) -> None:
    diagnostics.finish(
        "pretrain-step",
        source_complete=True,
        done=diagnostics.state["channels"]["pretrain-step"]["next"] == total + 1,
    )
    expected = total // cfg.reference_steps + 1
    diagnostics.finish(
        "pretrain-reference",
        source_complete=True,
        done=diagnostics.state["channels"]["pretrain-reference"]["next"] == expected,
    )
