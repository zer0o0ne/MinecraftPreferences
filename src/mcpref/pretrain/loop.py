"""(T) of specs/pretrain.md with ordinary, reference and selected diagnostics."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref import REPO
from mcpref.pretrain.checkpoints import (
    Modules,
    prune,
    restore,
    restore_rng,
    write_checkpoint,
    write_model,
)
from mcpref.pretrain.checks import (
    Groups,
    check_groups,
    check_inputs,
    given,
    header,
    index_run,
)
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.gradient_flow import accepted_records, gradient_dir, summarize
from mcpref.pretrain.items import Part, part, read_items
from mcpref.pretrain.reference import ReferenceData, write_metrics
from mcpref.pretrain.reference_eval import evaluate_reference
from mcpref.pretrain.selected_step import selected_step
from mcpref.pretrain.startup import (
    checkpoint_dir,
    reference_dir,
    start_gradient_flow,
    start_reference,
)
from mcpref.pretrain.step import accumulator, step
from mcpref.pretrain.update import adamw
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards
from mcpref.training.schedule import remaining
from mcpref.vpt.plan import generator

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}
LOGGED = ("phi", "pi", "P", "loss")
DONE = "pretrain: already done"


def started(
    cfg: PretrainConfig, modules: Modules, groups: Groups, h: int
) -> tuple[torch.optim.AdamW, dict]:
    """(T) 2: AdamW and the position, fresh (torch seeded) or from checkpoint h - 1."""
    if h == 0:
        torch.manual_seed(cfg.seed)
        return adamw(cfg, groups), dict(START)
    state = restore(checkpoint_dir(cfg), h, modules)
    optimizer = adamw(cfg, groups)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state, torch.device(cfg.device))
    return optimizer, {k: state[k] for k in START}


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
) -> None:
    """Schedule continuation with accepted selected records inside checkpoints."""
    directory, device = checkpoint_dir(cfg), torch.device(cfg.device)
    rng = generator([cfg.seed, 3], position["schedule_rng"])
    fit = reference.baselines if reference is not None else None
    windows = accumulator(cfg, fit)
    selected = {
        entry["step"]: (index, entry) for index, entry in enumerate(selection["steps"])
    }
    visit, t, losses, pending = None, position["step"], [], []
    for at, s, positions in remaining(rng, data.sizes, cfg.batch, position, cfg.epochs):
        if visit != (at["epoch"], s):
            visit, items = (at["epoch"], s), read_items(cfg, s)
        t += 1
        batch = [items[i] for i in positions]
        if t in selected:
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
        windows.add(entry, observations)
        if t % cfg.checkpoint_steps == 0 or t == data.total:
            saved = {"step": t} | at
            diagnostics = windows.drain(final=t == data.total)
            write_checkpoint(
                directory,
                j,
                saved,
                modules,
                optimizer,
                device,
                losses,
                diagnostics,
                pending,
            )
            prune(directory, j, cfg.keep_checkpoints)
            pending = []
            write_json(
                gradient_dir(cfg) / "summary.json",
                summarize(accepted_records(cfg), cfg.quantiles),
            )
            j, losses = j + 1, []
            if reference is not None and t % cfg.reference_steps == 0:
                metrics = evaluate_reference(cfg, modules, reference, run, t)
                write_metrics(reference_dir(cfg), t // cfg.reference_steps, metrics)
        bar.set_postfix({k: entry[k] for k in LOGGED if entry[k] is not None})
        bar.update()


def train(
    cfg: PretrainConfig,
    modules: Modules,
    groups: Groups,
    data: Part,
    run: dict,
) -> None:
    """Restore, initialize/reuse diagnostics, then continue the unchanged schedule."""
    for module in modules.values():
        module.train()
    directory = checkpoint_dir(cfg)
    h = complete_shards(directory)
    if h:
        prune(directory, h - 1, cfg.keep_checkpoints)
    optimizer, position = started(cfg, modules, groups, h)
    reference = start_reference(cfg, modules, run, position["step"])
    selection = start_gradient_flow(cfg, run, data)
    if position["step"] == data.total:
        print(DONE, file=sys.stderr)
        return
    with tqdm(
        total=data.total, initial=position["step"], desc="pretrain", unit="step"
    ) as bar:
        steps(
            cfg,
            modules,
            optimizer,
            groups,
            data,
            position,
            h,
            bar,
            reference,
            run,
            selection,
        )


def pretrain(
    cfg: PretrainConfig, path: Path, modules: dict[str, torch.nn.Module]
) -> None:
    """Behaviour: (R), then (T) unless model.pt exists; run.json complete at last."""
    run = index_run(cfg)
    modules = given(cfg, run, modules)
    groups = check_groups(cfg, modules)
    check_inputs(cfg, run)
    head = header(cfg, path)
    model = REPO / cfg.out_dir / "model.pt"
    if model.exists():
        print(DONE, file=sys.stderr)
    else:
        train(cfg, modules, groups, part(cfg, run), run)
        write_model(model, modules)
    write_json(REPO / cfg.out_dir / "run.json", head | {"complete": True})
