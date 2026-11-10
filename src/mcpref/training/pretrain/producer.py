"""Block 33 outbox staging and exact accepted-artifact joins for pretraining."""

import json
from collections.abc import Sequence

import torch

from mcpref.config import REPO
from mcpref.diagnostics.core.observations import (
    PolicyObservations as DiagnosticPolicyObservations,
)
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.diagnostics.core.types import ProducerContext, ReferencePair
from mcpref.diagnostics.preference.suites import GlobalFit, PreferenceRows
from mcpref.diagnostics.producers.policy import on_policy_observation
from mcpref.diagnostics.producers.preference import (
    join_pretrain_step,
    join_reference,
    on_pretrain_step,
    on_reference,
)
from mcpref.infrastructure.shards.store import complete_shards, name
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.references.reference import ReferenceData
from mcpref.training.pretrain.references.select import PARTS

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
    observations,
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
    record = on_pretrain_step(
        observations.preference, fit=fit, cfg=diagnostics.cfg, context=context
    )
    raw = getattr(observations, "raw_policy", ())
    if raw and (not step % cfg.checkpoint_steps or step == total):
        targets = torch.cat([item["targets"] for item in raw])
        valid = targets.sum(dim=-1) > 0
        prior = targets[valid].sum(dim=0)
        if prior.sum() > 0:
            prior = prior / prior.sum()
        observation = DiagnosticPolicyObservations(
            torch.cat([item["logits"] for item in raw]),
            targets,
            torch.cat([item["masks"] for item in raw]),
            torch.cat([item["previous"] for item in raw]),
            torch.cat([item["actions"] for item in raw]),
            prior,
            1,
            cfg.action_copy_eps,
            cfg.calibration_bins,
        )
        policy_clusters = tuple(cluster for item in raw for cluster in item["clusters"])
        policy_context = ProducerContext(
            "pretrain-policy",
            "train",
            "joint",
            "pretrain",
            cfg.out_dir,
            {},
            (step,),
            policy_clusters,
            tuple(sources),
        )
        policy_record = on_policy_observation(
            observation, cfg=diagnostics.cfg, context=policy_context
        )
        record["metrics"].update(policy_record["metrics"])
        record["populations"].update(policy_record["populations"])
        record["joined"]["policy"] = policy_record["joined"]
    diagnostics.stage("pretrain-step", step, f"step:{step}", step + 1, sources, record)


def accept_step(
    diagnostics: DiagnosticRun,
    step: int,
    *,
    save: bool = True,
    remove: bool = True,
    index: bool = True,
) -> None:
    accepted = diagnostics.accept(
        "pretrain-step",
        f"step:{step}",
        join=join_pretrain_step,
        save=False,
        remove=False,
    )
    if save and accepted:
        diagnostics.save(index=index)
    if remove:
        diagnostics.remove([path for path, _ in diagnostics.outbox("pretrain-step")])


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
