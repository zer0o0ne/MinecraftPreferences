"""Diagnostic startup, reuse and crash recovery of pretraining."""

import shutil
import sys
from pathlib import Path

from mcpref import REPO
from mcpref.pretrain.checkpoints import Modules
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.gradient_flow import (
    accepted_records,
    gradient_dir,
    select_gradient_steps,
    summarize,
)
from mcpref.pretrain.items import Part
from mcpref.pretrain.reference import (
    ReferenceData,
    load_initial,
    select_reference,
    write_initial,
    write_metrics,
)
from mcpref.pretrain.reference_eval import evaluate_initial, evaluate_reference
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards


def checkpoint_dir(cfg: PretrainConfig) -> Path:
    return REPO / cfg.out_dir / "checkpoints"


def reference_dir(cfg: PretrainConfig) -> Path:
    return REPO / cfg.out_dir / "reference"


def start_reference(
    cfg: PretrainConfig, modules: Modules, run: dict, restored_step: int
) -> ReferenceData | None:
    """(18), (21): initialize shard 0 or validate/repair only the latest due shard."""
    if not run["counts"]["train"]["pairs"]:
        print("pretrain reference: nothing to do", file=sys.stderr)
        return None
    directory = reference_dir(cfg)
    count = complete_shards(directory)
    if count == 0 and restored_step:
        raise RuntimeError("reference shard 000000 is missing after training started")
    selection, fit = select_reference(cfg, run)
    if count == 0:
        initial, metrics = evaluate_initial(cfg, modules, selection, fit, run)
        write_initial(directory, selection, fit, initial, metrics)
        return ReferenceData(selection, fit, initial["Z0"], initial["P0"], metrics)
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
        metrics = evaluate_reference(cfg, modules, reference, run, restored_step)
        write_metrics(directory, k, metrics)
        return reference
    raise RuntimeError("reference shards do not match the restored training step")


def start_gradient_flow(cfg: PretrainConfig, run: dict, data: Part) -> dict:
    """Discard unaccepted work, reuse/replay selection, and rebuild accepted summary."""
    root = gradient_dir(cfg)
    shutil.rmtree(root / "work", ignore_errors=True)
    selection = select_gradient_steps(cfg, run, data)
    write_json(root / "summary.json", summarize(accepted_records(cfg), cfg.quantiles))
    return selection
