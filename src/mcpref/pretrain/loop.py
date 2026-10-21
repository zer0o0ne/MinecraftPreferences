"""(T) and `pretrain` of specs/pretrain.md: after (R), (2)-(7) over the schedule (3),
continued from the last checkpoint; model.pt; run.json complete."""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
from mcpref.pretrain.clip import Clip, read_clip
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.loss import step_gradients
from mcpref.pretrain.update import adamw, update
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards, shard_path
from mcpref.training.schedule import n_batches, remaining
from mcpref.vpt.plan import generator

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}
LOGGED = ("phi", "pi", "P", "loss")
DONE = "pretrain: already done"


@dataclass(frozen=True)
class Part:
    """Notation: the training part of X: |I_s|, M, K_tot, R_tot, T; read(p) the clip."""

    sizes: list[int]
    m: int
    k_tot: int
    r_tot: int
    total: int
    read: Callable[[str], Clip]


def part(cfg: PretrainConfig, run: dict) -> Part:
    """Notation: |I_s| from N and B_X, M, K_tot = N_U + 2 N_R, R_tot = N_R, T = E M."""
    n_u, n_r = run["counts"]["train"]["clips"], run["counts"]["train"]["pairs"]
    b_x, n = run["config"]["items_per_shard"], n_u + n_r
    s = -(-n // b_x)
    sizes = [b_x] * (s - 1) + [n - (s - 1) * b_x]
    m = n_batches(sizes, cfg.batch)

    def read(p: str) -> Clip:
        return read_clip(REPO / p, run["states"], run["width"], run["height"])

    return Part(sizes, m, n_u + 2 * n_r, n_r, cfg.epochs * m, read)


def checkpoint_dir(cfg: PretrainConfig) -> Path:
    """Interface Checkpoints: {out_dir}/checkpoints/."""
    return REPO / cfg.out_dir / "checkpoints"


def read_items(cfg: PretrainConfig, s: int) -> list[dict]:
    """(T) 3: the items of shard s of {X}/train/."""
    path = shard_path(REPO / cfg.index_dir / "train", s) / "items.json"
    return json.loads(path.read_bytes())["items"]


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


def step(
    cfg: PretrainConfig,
    modules: Modules,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    data: Part,
    batch: list[dict],
    t: int,
) -> dict:
    """Step t on the mini-batch `batch`: (2) from gradients None, (4)-(6); its entry
    of losses.json (7)."""
    optimizer.zero_grad(set_to_none=True)
    logged = step_gradients(
        batch,
        modules,
        read=data.read,
        m=data.m,
        k_tot=data.k_tot,
        r_tot=data.r_tot,
        w_pi=cfg.w_pi,
        w_P=cfg.w_P,
        micro_batch=cfg.micro_batch,
    )
    return {"step": t} | logged | {"lr": update(cfg, optimizer, groups, t, data.total)}


def steps(
    cfg: PretrainConfig,
    modules: Modules,
    optimizer: torch.optim.AdamW,
    groups: Groups,
    data: Part,
    position: dict,
    j: int,
    bar: tqdm,
) -> None:
    """(T) 3: the steps after `position`, each shard's items read once per visit;
    checkpoint j, j + 1, ... when N_c | t or t = T, then pruned; the bar advanced."""
    directory, device = checkpoint_dir(cfg), torch.device(cfg.device)
    rng = generator([cfg.seed, 3], position["schedule_rng"])
    visit, t, losses = None, position["step"], []
    for at, s, positions in remaining(rng, data.sizes, cfg.batch, position, cfg.epochs):
        if visit != (at["epoch"], s):
            visit, items = (at["epoch"], s), read_items(cfg, s)
        t += 1
        batch = [items[i] for i in positions]
        entry = step(cfg, modules, optimizer, groups, data, batch, t)
        losses.append(entry)
        if t % cfg.checkpoint_steps == 0 or t == data.total:
            saved = {"step": t} | at
            write_checkpoint(directory, j, saved, modules, optimizer, device, losses)
            prune(directory, j, cfg.keep_checkpoints)
            j, losses = j + 1, []
        bar.set_postfix({k: entry[k] for k in LOGGED if entry[k] is not None})
        bar.update()


def train(cfg: PretrainConfig, modules: Modules, groups: Groups, data: Part) -> None:
    """(T) 1-3: train mode; the checkpoints pruned; the start; the steps (bar
    `pretrain`, or its line after a restored step T)."""
    for module in modules.values():
        module.train()
    directory = checkpoint_dir(cfg)
    h = complete_shards(directory)
    if h:
        prune(directory, h - 1, cfg.keep_checkpoints)
    optimizer, position = started(cfg, modules, groups, h)
    if position["step"] == data.total:
        print(DONE, file=sys.stderr)
        return
    with tqdm(
        total=data.total, initial=position["step"], desc="pretrain", unit="step"
    ) as bar:
        steps(cfg, modules, optimizer, groups, data, position, h, bar)


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
        train(cfg, modules, groups, part(cfg, run))
        write_model(model, modules)
    write_json(REPO / cfg.out_dir / "run.json", head | {"complete": True})
