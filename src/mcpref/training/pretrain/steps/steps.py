"""Scheduled optimizer steps and their accepted diagnostic events."""

import torch
from tqdm import tqdm

from mcpref.datasets.vpt.plan import generator
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.infrastructure.profiling.runtime import TrainingProfile
from mcpref.infrastructure.profiling.scopes import phase
from mcpref.infrastructure.runs.files import write_json
from mcpref.training.checkpoints import Modules, prune
from mcpref.training.groups import Groups
from mcpref.training.pretrain.checkpoints import write_checkpoint
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import Part, read_items
from mcpref.training.pretrain.gradients.flow import (
    accepted_records,
    gradient_dir,
    summarize,
)
from mcpref.training.pretrain.producer import (
    accept_reference,
    accept_step,
    stage_reference,
    stage_step,
)
from mcpref.training.pretrain.references.conditions import evaluate_reference_event
from mcpref.training.pretrain.references.reference import ReferenceData, write_metrics
from mcpref.training.pretrain.startup import checkpoint_dir, reference_dir
from mcpref.training.pretrain.steps.selected_step import selected_step
from mcpref.training.pretrain.steps.step import accumulator, step
from mcpref.training.schedule import remaining

LOGGED = ("phi", "pi", "P", "loss")


def steps(
    cfg: PretrainConfig,
    modules: Modules,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    data: Part,
    position: dict,
    j: int,
    bar: tqdm,
    reference: ReferenceData | None,
    run: dict,
    selection: dict,
    diagnostics: DiagnosticRun,
    diagnostic: tqdm | None,
) -> None:
    directory, device = checkpoint_dir(cfg), torch.device(cfg.device)
    rng = generator([cfg.seed, 3], position["schedule_rng"])
    fit = reference.baselines if reference is not None else None
    windows = accumulator(cfg, fit)
    selected = {
        entry["step"]: (index, entry) for index, entry in enumerate(selection["steps"])
    }
    visit, t, losses, pending = None, position["step"], [], []
    with TrainingProfile(cfg, directory.parent, modules, "pretrain") as profile:
        for at, s, positions in remaining(
            rng, data.sizes, cfg.batch, position, cfg.epochs
        ):
            with profile.step(t + 1, len(positions)):
                with phase("data.index_read"):
                    if visit != (at["epoch"], s):
                        visit, items = (at["epoch"], s), read_items(cfg, s)
                t += 1
                batch = [items[i] for i in positions]
                if cfg.gradient_diagnostics and t in selected:
                    index, chosen = selected[t]
                    entry, observations, record = selected_step(
                        cfg,
                        modules,
                        optimizer,
                        groups,
                        data,
                        batch,
                        t,
                        fit,
                        run,
                        chosen,
                        index,
                        j,
                        len(losses),
                    )
                    pending.append(record)
                else:
                    entry, observations = step(
                        cfg, modules, optimizer, groups, data, batch, t, fit
                    )
                losses.append(entry)
                with phase("diagnostics.window"):
                    windows.add(entry, observations)
                if fit is not None:
                    stage_step(
                        diagnostics,
                        cfg,
                        fit,
                        j,
                        t,
                        data.total,
                        batch,
                        observations,
                    )
                if t % cfg.checkpoint_steps == 0 or t == data.total:
                    saved = {"step": t} | at
                    window_records = windows.drain(final=t == data.total)
                    write_checkpoint(
                        directory,
                        j,
                        saved,
                        modules,
                        optimizer,
                        device,
                        losses,
                        window_records,
                        pending,
                    )
                    prune(directory, j, cfg.keep_checkpoints)
                    pending = []
                    if cfg.gradient_diagnostics:
                        write_json(
                            gradient_dir(cfg) / "summary.json",
                            summarize(accepted_records(cfg), cfg.quantiles),
                        )
                    if fit is not None:
                        for loss in losses:
                            accept_step(
                                diagnostics, loss["step"], remove=False, index=False
                            )
                            if diagnostic is not None:
                                diagnostic.update()
                        diagnostics.save()
                        diagnostics.remove(
                            [path for path, _ in diagnostics.outbox("pretrain-step")]
                        )
                    j, losses = j + 1, []
                    if reference is not None and t % cfg.reference_steps == 0:
                        k = t // cfg.reference_steps
                        metrics, pairs, observations = evaluate_reference_event(
                            cfg, modules, reference, run, t
                        )
                        stage_reference(
                            diagnostics, cfg, reference, k, pairs, observations
                        )
                        write_metrics(reference_dir(cfg), k, metrics, pairs)
                        accept_reference(diagnostics, k)
                        if diagnostic is not None:
                            diagnostic.update()
                bar.set_postfix({k: entry[k] for k in LOGGED if entry[k] is not None})
                bar.update()
