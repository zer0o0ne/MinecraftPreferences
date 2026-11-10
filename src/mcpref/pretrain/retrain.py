"""Optional frozen-phi retraining of pi and P with exact resume."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

from mcpref import REPO
from mcpref.pretrain.checkpoints import (
    prune,
    restore,
    restore_rng,
    write_checkpoint,
    write_model,
)
from mcpref.pretrain.checks import Groups, check_groups
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.items import Part, read_items
from mcpref.pretrain.reference import load_global_fit
from mcpref.pretrain.step import accumulator, step
from mcpref.pretrain.update import adamw
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards
from mcpref.training.schedule import remaining
from mcpref.vpt.plan import generator

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}


def head_groups(cfg: PretrainConfig, modules: dict) -> Groups:
    declared = check_groups(cfg, modules)
    heads = {
        id(parameter)
        for name in ("pi", "P")
        if name in modules
        for parameter in modules[name].parameters()
    }
    selected = Groups()
    selected.parameter_names = declared.parameter_names
    for name, parameters in declared.items():
        group = [parameter for parameter in parameters if id(parameter) in heads]
        if group:
            selected[name] = group
    return selected


def load_joint_model(path: Path, modules: dict) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(state[name])


def retrain(cfg: PretrainConfig, modules: dict, data: Part) -> None:
    root = REPO / cfg.out_dir
    marker = root / "retrain/run.json"
    if marker.exists():
        return
    directory = root / "retrain/checkpoints"
    h = complete_shards(directory)
    groups = head_groups(cfg, modules)
    if h == 0:
        load_joint_model(root / "model.pt", modules)
        torch.manual_seed(cfg.seed)
        optimizer = adamw(cfg, groups)
        position = dict(START)
    else:
        state = restore(directory, h, modules)
        optimizer = adamw(cfg, groups)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state, torch.device(cfg.device))
        position = {key: state[key] for key in START}
    modules["phi"].requires_grad_(False)
    modules["phi_loss"].requires_grad_(False)
    for module in modules.values():
        module.train()
    fit = load_global_fit(root) if data.r_tot else None
    windows = accumulator(cfg, fit)
    visit, t, losses, j = None, position["step"], [], h
    with tqdm(
        total=data.total,
        initial=t,
        desc="pretrain retrain",
        unit="step",
        file=sys.stderr,
    ) as bar:
        for at, shard, positions in remaining(
            generator([cfg.seed, 3], position["schedule_rng"]),
            data.sizes,
            cfg.batch,
            position,
            cfg.epochs,
        ):
            if visit != (at["epoch"], shard):
                visit, items = (at["epoch"], shard), read_items(cfg, shard)
            t += 1
            batch = [items[i] for i in positions]
            entry, observations = step(
                cfg, modules, optimizer, groups, data, batch, t, fit
            )
            losses.append(entry)
            windows.add(entry, observations)
            if t % cfg.checkpoint_steps == 0 or t == data.total:
                write_checkpoint(
                    directory,
                    j,
                    {"step": t} | at,
                    modules,
                    optimizer,
                    torch.device(cfg.device),
                    losses,
                    windows.drain(final=t == data.total),
                    [],
                )
                prune(directory, j, cfg.keep_checkpoints)
                j, losses = j + 1, []
            bar.set_postfix({key: entry[key] for key in ("phi", "pi", "P", "loss")})
            bar.update()
    write_model(root / "model.pt", modules)
    write_json(marker, {"complete": True, "steps": data.total})
