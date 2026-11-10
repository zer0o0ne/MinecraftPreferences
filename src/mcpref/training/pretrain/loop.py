"""(T) of specs/pretrain.md with ordinary, reference and selected diagnostics."""

import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref.config import REPO
from mcpref.diagnostics.core.runtime import DiagnosticRun, diagnostic_bar
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards
from mcpref.training.checkpoints import Modules, prune, restore_rng, write_model
from mcpref.training.groups import Groups, check_groups
from mcpref.training.optim import adamw
from mcpref.training.pretrain.checkpoints import restore
from mcpref.training.pretrain.checks import check_inputs, given, header, index_run
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import Part, part
from mcpref.training.pretrain.producer import finish, register
from mcpref.training.pretrain.references.reference import load_initial
from mcpref.training.pretrain.startup import (
    checkpoint_dir,
    reference_dir,
    start_gradient_flow,
    start_reference,
)
from mcpref.training.pretrain.steps.steps import steps

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}
DONE = "pretrain: already done"


def started(
    cfg: PretrainConfig, modules: Modules, groups: Groups, h: int
) -> tuple[torch.optim.AdamW, dict]:
    if h == 0:
        torch.manual_seed(cfg.seed)
        return adamw(cfg, groups), dict(START)
    state = restore(checkpoint_dir(cfg), h, modules)
    optimizer = adamw(cfg, groups)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state, torch.device(cfg.device))
    return optimizer, {k: state[k] for k in START}


def diagnostic_progress(cfg: PretrainConfig, data: Part, diagnostics: DiagnosticRun):
    step_state = diagnostics.state["channels"]["pretrain-step"]
    reference_state = diagnostics.state["channels"]["pretrain-reference"]
    reference_total = data.total // cfg.reference_steps + 1
    total = (
        data.total
        - step_state["watermark"]
        + 1
        + reference_total
        - reference_state["watermark"]
    )
    initial = (
        step_state["next"]
        - step_state["watermark"]
        + reference_state["next"]
        - reference_state["watermark"]
    )
    return diagnostic_bar("diagnostics pretrain", total, initial, "step")


def train(
    cfg: PretrainConfig,
    modules: Modules,
    groups: Groups,
    data: Part,
    run: dict,
    diagnostics: DiagnosticRun,
) -> None:
    for module in modules.values():
        module.train()
    directory = checkpoint_dir(cfg)
    h = complete_shards(directory)
    if h:
        prune(directory, h - 1, cfg.keep_checkpoints)
    optimizer, position = started(cfg, modules, groups, h)
    instrumented = bool(run["counts"]["train"]["pairs"])
    if instrumented:
        register(cfg, diagnostics, position["step"])
        progress = diagnostic_progress(cfg, data, diagnostics)
    else:
        progress = nullcontext(None)
    with progress as diagnostic:
        reference = start_reference(
            cfg, modules, run, position["step"], diagnostics, diagnostic
        )
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
                diagnostics,
                diagnostic,
            )


def pretrain(
    cfg: PretrainConfig, path: Path, modules: dict[str, torch.nn.Module]
) -> None:
    run = index_run(cfg)
    modules = given(cfg, run, modules)
    groups = check_groups(cfg, modules)
    check_inputs(cfg, run)
    head = header(cfg, path)
    diagnostics = DiagnosticRun.load(REPO / cfg.diagnostics)
    data = part(cfg, run)
    model = REPO / cfg.out_dir / "model.pt"
    completed = model.exists()
    if completed:
        print(DONE, file=sys.stderr)
    else:
        train(cfg, modules, groups, data, run, diagnostics)
        write_model(model, modules)
    if cfg.retrain:
        from mcpref.training.pretrain.retrain import retrain

        retrain(cfg, modules, data)
    write_json(REPO / cfg.out_dir / "run.json", head | {"complete": True})
    if run["counts"]["train"]["pairs"]:
        if completed:
            register(cfg, diagnostics, data.total)
            load_initial(reference_dir(cfg))
            with diagnostic_progress(cfg, data, diagnostics):
                pass
        finish(data.total, cfg, diagnostics)
