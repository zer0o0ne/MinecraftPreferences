"""Transcription of specs/pretrain.md (1)-(7) in float64, nothing from src/: the terms of (1)
over a run of items (the gradients by autograd through the fake components of
loop_modules.py, called in the order of (2)), the micro-batches of (2), the schedule (3)
with `Generator(PCG64([seed, 3]))`, the clipping (4), the learning rates (5), AdamW (6) in
numpy and the logged losses (7); from `torch.manual_seed(seed)` on fresh components."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from training.schedule_oracle import epoch, mini_batches

from .loop_modules import fakes, group_order
from .loop_world import World, clip


def terms(items, mods, read, m, k_tot, r_tot, w_pi, w_P) -> tuple:
    """(1) over the clips K and the pairs R of `items`: (the loss, l_phi, l_pi, l_P or
    None), the calls in the order phi, l_phi, pi, l_pi, P, l_P."""
    clips, at, labels = [], [], []
    for it in items:
        if "clip" in it:
            clips.append(read(it["clip"]))
        else:
            at.append((len(clips), len(clips) + 1))
            clips += [read(it["a"]), read(it["b"])]
            labels.append(it["labels"])
    Z = mods["phi"](clips)
    l_phi = mods["phi_loss"](Z, clips)
    l_pi = mods["pi_loss"](mods["pi"](Z), Z, clips)
    loss = m / k_tot * (l_phi + w_pi * l_pi).sum()
    l_P = None
    if labels:
        ia, ib = (list(x) for x in zip(*at))
        l_P = mods["P_loss"](mods["P"](Z[ia], Z[ib]), labels)
        loss = loss + m * w_P / r_tot * l_P.sum()
    return loss, l_phi, l_pi, l_P


def gradients(items, mods, read, m, k_tot, r_tot, w_pi, w_P, micro) -> dict:
    """(2): backward of each micro-batch's terms, accumulating; (7) without lr."""
    sums, n_k, n_r = {"phi": 0.0, "pi": 0.0, "P": 0.0, "loss": 0.0}, 0, 0
    for j in range(0, len(items), micro):
        loss, l_phi, l_pi, l_P = terms(items[j : j + micro], mods, read, m, k_tot, r_tot, w_pi, w_P)  # fmt: skip
        loss.backward()
        sums["phi"] += l_phi.detach().sum().item()
        sums["pi"] += l_pi.detach().sum().item()
        sums["loss"] += loss.detach().item()
        n_k += len(l_phi)
        if l_P is not None:
            sums["P"] += l_P.detach().sum().item()
            n_r += len(l_P)
    return {"phi": sums["phi"] / n_k, "pi": sums["pi"] / n_k,
            "P": sums["P"] / n_r if n_r else None, "loss": sums["loss"]}  # fmt: skip


def factor(t: int, T: int, tw: int, rho: float) -> float:
    """(5): f(t)."""
    if t <= tw:
        return t / tw
    return rho + (1 - rho) * 0.5 * (1 + math.cos(math.pi * (t - tw) / (T - tw)))


@dataclass
class Moments:
    n: int
    m: np.ndarray
    v: np.ndarray


@dataclass
class Step:
    """The state after step t."""

    t: int
    epoch: int
    batch_in_epoch: int
    losses: dict  # the entry of losses.json
    modules: dict  # name -> state_dict (copies)
    adam: list  # per parameter in group order: Moments, None before its first update
    torch_rng: torch.Tensor
    clipped: bool


def adamw(p: torch.Tensor, g: np.ndarray, st: Moments, lr, wd, betas, eps) -> None:
    """(6) on one parameter whose gradient g is not None."""
    (b1, b2), x = betas, p.detach().numpy().astype(np.float64)
    st.n += 1
    x = x * (1 - lr * wd)
    st.m = b1 * st.m + (1 - b1) * g
    st.v = b2 * st.v + (1 - b2) * g * g
    corr = math.sqrt(1 - b2**st.n)
    x = x - lr * (st.m / (1 - b1**st.n)) / (np.sqrt(st.v) / corr + eps)
    with torch.no_grad():
        p.copy_(torch.from_numpy(np.asarray(x)))


def update(groups, moments, cfg: dict, f: float) -> bool:
    """(4), (5), (6) after the backward passes of a step; whether the gradients were
    scaled by a factor below 1."""
    grads = [[None if q.grad is None else q.grad.numpy().astype(np.float64) for q in ps]
             for _, ps in groups]  # fmt: skip
    norm = math.sqrt(
        sum(float((g**2).sum()) for gs in grads for g in gs if g is not None)
    )
    coef = min(cfg["grad_clip"] / (norm + 1e-6), 1.0)
    for (q, ps), gs, sts in zip(groups, grads, moments):
        hyper = cfg["groups"][q]
        for i, (p, g) in enumerate(zip(ps, gs)):
            if g is None:
                continue
            sts[i] = sts[i] or Moments(0, np.zeros_like(g), np.zeros_like(g))
            lr = hyper["lr"] * f
            adamw(p, g * coef, sts[i], lr, hyper["weight_decay"], cfg["betas"], cfg["adam_eps"])  # fmt: skip
    return coef < 1.0


def train(world: World, cfg: dict, make=None) -> tuple[list[Step], list]:
    """The steps of a run from fresh components (`make()`, default the fakes), and the
    schedule generator's state before the draws of each epoch."""
    mods = make() if make else fakes(world.n_r > 0)
    groups = group_order(mods)
    moments = [[None] * len(ps) for _, ps in groups]
    sizes, b = [len(s) for s in world.train], cfg["batch"]
    M = mini_batches(sizes, b)
    T, k_tot, r_tot = cfg["epochs"] * M, world.n_u + 2 * world.n_r, world.n_r
    torch.manual_seed(cfg["seed"])
    g = np.random.Generator(np.random.PCG64([cfg["seed"], 3]))
    steps, states = [], []
    for e in range(1, cfg["epochs"] + 1):
        states.append(g.bit_generator.state)
        for i, (s, pos) in enumerate(epoch(g, sizes, b), 1):
            t = len(steps) + 1
            for _, ps in groups:
                for q in ps:
                    q.grad = None
            items = [world.train[s][k] for k in pos]
            logged = gradients(items, mods, clip, M, k_tot, r_tot, cfg["w_pi"],
                               cfg["w_P"], cfg["micro_batch"])  # fmt: skip
            f = factor(t, T, cfg["warmup_steps"], cfg["lr_floor"])
            clipped = update(groups, moments, cfg, f)
            lr = {q: cfg["groups"][q]["lr"] * f for q, _ in groups}
            snapshot = {k: {n: v.detach().clone() for n, v in m.state_dict().items()}
                        for k, m in mods.items()}  # fmt: skip
            adam = [Moments(st.n, st.m.copy(), st.v.copy()) if st else None
                    for sts in moments for st in sts]  # fmt: skip
            losses = {"step": t, **logged, "lr": lr}
            steps.append(Step(t, e, i, losses, snapshot, adam, torch.get_rng_state(), clipped))  # fmt: skip
    return steps, states
