"""(4), (F) of specs/gui-stack.md: the loss (3) over all of S and Adam over the schedule
of training/schedule.py, continued from the last checkpoint."""

import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mcpref.shards.store import complete_shards, shard_path, write_shard
from mcpref.stacking.config import GuiStackConfig
from mcpref.stacking.model import init_params, loss, penalty, weighted_nll
from mcpref.training.devices import cpu_copies
from mcpref.training.schedule import n_batches, remaining
from mcpref.vpt.plan import generator

Shard = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""z, y (label indices in Y), w_f of the frames of a shard of S."""
Read = Callable[[int], Shard]
Fitted = tuple[list[torch.Tensor], float, float]
"""theta_T, L(theta_0), L(theta_T)."""
CHECKPOINT = "checkpoint.pt"


def full_loss(
    params: Sequence[torch.Tensor], read: Read, q: int, l2: float, desc: str
) -> float:
    """(3) over all of S: sum w_f (-log p) and sum w_f accumulated over its q shards,
    read one at a time; bar `desc`."""
    nll, w_sum = 0.0, 0.0
    with torch.no_grad(), tqdm(total=q, desc=desc, unit="shard") as bar:
        for r in range(q):
            z, y, w = read(r)
            nll += weighted_nll(params, z, y, w).item()
            w_sum += w.sum().item()
            bar.update()
        return nll / w_sum + penalty(params, l2).item()


def adam(params: Sequence[torch.Tensor], cfg: GuiStackConfig) -> torch.optim.Adam:
    """(4): Adam with the config's eta, betas, epsilon; no weight decay, no AMSGrad."""
    return torch.optim.Adam(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.adam_eps,
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        decoupled_weight_decay=False,
    )


def started(cfg: GuiStackConfig, read: Read, q: int, c: int, n_labels: int) -> dict:
    """(2), (F): theta_0, L(theta_0) (bar `loss initial`), the start of epoch 1 (a fresh
    schedule generator)."""
    params = init_params(c, cfg.hidden, n_labels, cfg.seed, cfg.device)
    initial = full_loss(params, read, q, cfg.l2, "loss initial")
    position = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}
    return {"params": params, "loss_initial": initial} | position


def restored(directory: Path, h: int, device: torch.device) -> dict:
    """Interface checkpoints: the last of the h checkpoints, parameters on `device`."""
    state = torch.load(shard_path(directory, h - 1) / CHECKPOINT)
    state["params"] = [p.to(device).requires_grad_() for p in state["params"]]
    print("loss initial: already done", file=sys.stderr)
    return state


def adam_step(
    optimizer: torch.optim.Adam, objective: Callable, data: Shard, positions: np.ndarray
) -> None:
    """(4): one Adam update on the gradient of `objective` (3') on the mini-batch
    `positions` of a shard's (z, y, w)."""
    index = torch.from_numpy(positions).to(data[0].device)
    optimizer.zero_grad()
    objective(*(a[index] for a in data)).backward()
    optimizer.step()


def write_checkpoint(
    directory: Path, j: int, state: dict, optimizer: torch.optim.Adam, position: dict
) -> None:
    """Interface checkpoints/<name(j)>/checkpoint.pt: step, params, adam, loss_initial,
    epoch, schedule_rng, batch_in_epoch."""
    saved = {"params": cpu_copies(state["params"]), "adam": optimizer.state_dict()}
    saved |= {"loss_initial": state["loss_initial"]} | position
    with write_shard(directory, j) as staged:
        torch.save(saved, staged / CHECKPOINT)


def train(
    cfg: GuiStackConfig,
    read: Read,
    sizes: Sequence[int],
    w_total: float,
    state: dict,
    optimizer: torch.optim.Adam,
    directory: Path,
    update: Callable[[], object],
) -> None:
    """(4) from the position of `state`: each shard read once per visit; a checkpoint
    after every update whose step is a multiple of N, before `update()`."""
    rng = generator(cfg.seed, state["schedule_rng"])
    m = n_batches(sizes, cfg.batch)
    objective = partial(loss, state["params"], w_total=w_total, m=m, l2=cfg.l2)
    visit, data, t = None, None, state["step"]
    for position, q, positions in remaining(rng, sizes, cfg.batch, state, cfg.epochs):
        if visit != (position["epoch"], q):
            visit, data = (position["epoch"], q), read(q)
        adam_step(optimizer, objective, data, positions)
        t += 1
        if t % cfg.checkpoint_steps == 0:
            j = t // cfg.checkpoint_steps - 1
            write_checkpoint(directory, j, state, optimizer, {"step": t} | position)
        update()


def fit(
    cfg: GuiStackConfig,
    read: Read,
    sizes: Sequence[int],
    c: int,
    n_labels: int,
    w_total: float,
    directory: Path,
) -> Fitted:
    """(2)-(4) continued from the last checkpoint of `directory`: theta_T, L(theta_0),
    L(theta_T); bars `loss initial`, `stack fit`, `loss final`."""
    h = complete_shards(directory)
    if h:
        state = restored(directory, h, cfg.device)
    else:
        state = started(cfg, read, len(sizes), c, n_labels)
    optimizer = adam(state["params"], cfg)
    if h:
        optimizer.load_state_dict(state["adam"])
    total = cfg.epochs * n_batches(sizes, cfg.batch)
    if state["step"] == total:
        print("stack fit: already done", file=sys.stderr)
    else:
        with tqdm(
            total=total, initial=state["step"], desc="stack fit", unit="step"
        ) as bar:
            train(cfg, read, sizes, w_total, state, optimizer, directory, bar.update)
    final = full_loss(state["params"], read, len(sizes), cfg.l2, "loss final")
    return state["params"], state["loss_initial"], final
