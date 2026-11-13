"""Diagnostic startup, reuse and crash recovery of pretraining."""

import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import Part
from mcpref.training.pretrain.gradients.flow import (
    accepted_records,
    gradient_dir,
    select_gradient_steps,
    summarize,
)
from mcpref.training.pretrain.producer import accept_reference, stage_reference
from mcpref.training.pretrain.references.conditions import (
    evaluate_initial_event,
    evaluate_reference_event,
)
from mcpref.training.pretrain.references.reference import (
    ReferenceData,
    load_initial,
    select_reference,
    write_initial,
    write_metrics,
)


def checkpoint_dir(cfg: PretrainConfig) -> Path:
    return REPO / cfg.out_dir / "checkpoints"


def reference_dir(cfg: PretrainConfig) -> Path:
    return REPO / cfg.out_dir / "reference"


def start_reference(
    cfg: PretrainConfig,
    modules: Modules,
    run: dict,
    restored_step: int,
    diagnostics: DiagnosticRun,
    diagnostic: tqdm | None,
) -> ReferenceData | None:
    """Initialize or repair only the latest due shard, accepting its persisted outbox."""
    if not run["counts"]["train"]["pairs"]:
        print("pretrain reference: nothing to do", file=sys.stderr)
        return None
    directory = reference_dir(cfg)
    count = complete_shards(directory)
    if count == 0 and restored_step:
        raise RuntimeError("reference shard 000000 is missing after training started")
    selection, fit = select_reference(cfg, run)
    if count == 0:
        initial, metrics, pairs, observations = evaluate_initial_event(
            cfg, modules, selection, fit, run
        )
        reference = ReferenceData(
            selection, fit, initial["Z0"], initial["P0"], metrics, tuple(pairs)
        )
        stage_reference(diagnostics, cfg, reference, 0, pairs, observations)
        write_initial(directory, selection, fit, initial, metrics, pairs)
        accept_reference(diagnostics, 0)
        if diagnostic is not None:
            diagnostic.update()
        return reference
    reference = load_initial(directory)
    print("pretrain reference initial: already done", file=sys.stderr)
    expected = restored_step // cfg.reference_steps + 1
    due = restored_step > 0 and restored_step % cfg.reference_steps == 0
    if count == expected:
        if due:
            print("pretrain reference: already done", file=sys.stderr)
        return reference
    if due and count == expected - 1:
        k = restored_step // cfg.reference_steps
        metrics, pairs, observations = evaluate_reference_event(
            cfg, modules, reference, run, restored_step
        )
        stage_reference(diagnostics, cfg, reference, k, pairs, observations)
        write_metrics(directory, k, metrics, pairs)
        accept_reference(diagnostics, k)
        if diagnostic is not None:
            diagnostic.update()
        return reference
    raise RuntimeError("reference shards do not match the restored training step")


def start_gradient_flow(cfg: PretrainConfig, run: dict, data: Part) -> dict:
    if not cfg.gradient_diagnostics:
        print("pretrain gradient flow: disabled", file=sys.stderr)
        return {"version": 1, "middle_target": max(1, data.total // 2), "steps": []}
    root = gradient_dir(cfg)
    shutil.rmtree(root / "work", ignore_errors=True)
    selection = select_gradient_steps(cfg, run, data)
    write_json(root / "summary.json", summarize(accepted_records(cfg), cfg.quantiles))
    return selection
