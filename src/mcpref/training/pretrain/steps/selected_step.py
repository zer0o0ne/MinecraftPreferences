"""One selected real optimizer step and its accepted-candidate record."""

import copy
import shutil
import sys

import torch
from tqdm import tqdm

from mcpref.diagnostics.pretrain.diagnostics import GlobalFit, StepObservations
from mcpref.infrastructure.shards.store import name
from mcpref.training.checkpoints import Modules
from mcpref.training.groups import Groups
from mcpref.training.optim import update
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import Part
from mcpref.training.pretrain.gradients.flow import (
    GradientStepRecord,
    baseline_pairs,
    counterfactual_metrics,
    gradient_dir,
    labelled_items,
    merge_alignment,
    pair_records,
    post_step_pairs,
)
from mcpref.training.pretrain.gradients.work import (
    contribution_count,
    reduce_work_with_progress,
    work_manifests,
)
from mcpref.training.pretrain.steps.loss import selected_step_gradients
from mcpref.training.pretrain.steps.step import step_record
from mcpref.training.pretrain.update import gradient_stats, update_and_align


def selected_step(
    cfg: PretrainConfig,
    modules: Modules,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    data: Part,
    batch: list[dict],
    t: int,
    fit: GlobalFit | None,
    run: dict,
    selected: dict,
    selection_index: int,
    checkpoint: int,
    losses_index: int,
) -> tuple[dict, StepObservations, GradientStepRecord]:
    """A real selected step with the unchanged update and diagnostic-only work."""
    optimizer.zero_grad(set_to_none=True)
    work = gradient_dir(cfg) / "work" / name(t)
    with tqdm(
        total=-(-len(batch) // cfg.micro_batch),
        desc="pretrain gradient flow collect",
        unit="micro-batch",
        file=sys.stderr,
        mininterval=0,
    ) as collect_bar:
        collect_bar.refresh()
        logged, micro_batches = selected_step_gradients(
            batch,
            modules,
            read=data.read,
            m=data.m,
            k_tot=data.k_tot,
            r_tot=data.r_tot,
            w_phi=cfg.w_phi,
            w_pi=cfg.w_pi,
            w_P=cfg.w_P,
            micro_batch=cfg.micro_batch,
            cfg=cfg,
            groups=groups,
            directory=work,
            progress=collect_bar,
        )
    logged["step"] = t
    observations = logged["observations"]
    gradients = gradient_stats(groups, cfg.grad_clip, preference=modules.get("P"))
    manifests = work_manifests(work)
    contributions = contribution_count(manifests) + contribution_count(manifests, "P")
    with tqdm(
        total=contributions,
        desc="pretrain gradient flow reduce",
        unit="contribution",
        file=sys.stderr,
        mininterval=0,
    ) as reduce_bar:
        reduce_bar.refresh()
        reduced = reduce_work_with_progress(work, groups, reduce_bar)
        frozen = copy.deepcopy(modules["P"])
        frozen.eval()
        labelled = labelled_items(batch, cfg.micro_batch)
        raw_items = [item for _, _, item in labelled]
        with tqdm(
            total=2 * len(raw_items),
            desc="pretrain gradient flow pairs",
            unit="evaluation",
            file=sys.stderr,
            mininterval=0,
        ) as pair_bar:
            pair_bar.refresh()
            baseline = baseline_pairs(cfg, modules, raw_items, run, pair_bar)
            lr = update(cfg, optimizer, groups, t, data.total)
            updates, alignment = update_and_align(
                cfg, optimizer, groups, lr, work, reduce_bar
            )
            optimizer.zero_grad(set_to_none=True)
            shutil.rmtree(work)
            work.parent.rmdir()
            only_phi, joint = post_step_pairs(
                cfg, modules, frozen, raw_items, run, pair_bar
            )
    reduced = merge_alignment(reduced, alignment)
    pairs = pair_records(labelled, baseline, only_phi, joint)
    shard = f"checkpoints/{name(checkpoint)}"
    pair_path = f"{shard}/gradient_flow/{name(selection_index)}/pairs.json"
    metrics = {
        "version": 1,
        "step": t,
        "roles": selected["roles"],
        "checkpoint_losses_index": losses_index,
        "micro_batches": micro_batches,
        "phi": reduced["phi"],
        "P": reduced["P"],
        "nonfinite": reduced["nonfinite"],
        "counterfactual": counterfactual_metrics(pairs, cfg.quantiles),
        "evidence": {
            "kind": "real_pretrain_step",
            "checkpoint_shard": shard,
            "pairs": pair_path,
        },
    }
    entry = step_record(cfg, fit, logged, observations, gradients, lr, updates)
    return entry, observations, GradientStepRecord(metrics, pairs)
