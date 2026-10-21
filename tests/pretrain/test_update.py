"""(4)-(6) of specs/pretrain.md through `pretrain`, a checkpoint after every step: each step's
clipping, learning rates and AdamW update against the oracle (loop_oracle.py); the learning
rates of (5) on both branches, $T_w = 0$ and $T_w \\ge T$; a parameter whose gradient is None
untouched, one whose gradient is zero decayed (hand formulas)."""

import math

import pytest
import torch

from .loop_checks import assert_run
from .loop_modules import fakes, group_order
from .loop_oracle import train
from .loop_run import Loop

EVERY_STEP = {"checkpoint_steps": 1, "keep_checkpoints": 20}  # T = 10, ten checkpoints


@pytest.mark.parametrize("clip", [1.0, 1e-3, 1e9], ids=["both", "always", "never"])
def test_every_step_against_oracle(tmp_path, clip):
    """(4), (5), (6) of each step: the components and AdamW's state after it."""
    loop = Loop(tmp_path, grad_clip=clip, **EVERY_STEP).write()
    loop.run()
    steps, states = train(loop.world, loop.config)
    want = {1.0: {True, False}, 1e-3: {True}, 1e9: {False}}[clip]
    assert {s.clipped for s in steps} == want
    assert_run(loop, steps, states, kept=20)


def f_hand(t: int, T: int, tw: int, rho: float) -> float:
    """(5) by hand: t / T_w up to T_w, then rho + (1 - rho)(1 + cos(pi (t - T_w) /
    (T - T_w))) / 2."""
    if t <= tw:
        return t / tw
    return rho + (1 - rho) * (1 + math.cos(math.pi * (t - tw) / (T - tw))) / 2


@pytest.mark.parametrize("tw", [0, 3, 10, 14])
def test_learning_rates(tmp_path, tw):
    """lr of every entry of losses.json and AdamW's lr after each step: eta_q f(t), T = 10;
    f(T_w) = 1, f(T) = rho for T_w < T; f(t) = t / T_w for T_w >= T."""
    loop = Loop(tmp_path, warmup_steps=tw, **EVERY_STEP).write()
    loop.run()
    groups = [q for q, _ in group_order(fakes())]
    rho, T = loop.config["lr_floor"], 10
    for t in range(1, T + 1):
        (entry,) = loop.losses(t - 1)
        want = {
            q: loop.config["groups"][q]["lr"] * f_hand(t, T, tw, rho) for q in groups
        }
        assert list(entry["lr"]) == groups
        assert entry["lr"] == pytest.approx(want, rel=1e-12)
        got = [pg["lr"] for pg in loop.checkpoint(t - 1)["optimizer"]["param_groups"]]
        assert got == pytest.approx([want[q] for q in groups], rel=1e-12)
    if tw < T:
        assert f_hand(T, T, tw, rho) == pytest.approx(rho)
    if 1 <= tw <= T:
        assert f_hand(tw, T, tw, rho) == 1


def p_tensors(ck: dict) -> list:
    return [ck["modules"]["P"][k] for k in ("u", "beta", "v")]


def indices(group: str) -> list[int]:
    """The indices in AdamW's state_dict of the parameters of `group`."""
    at = [q for q, ps in group_order(fakes()) for _ in ps]
    return [i for i, q in enumerate(at) if q == group]


def test_parameter_with_gradient_none_untouched(tmp_path):
    """A step without a pair (P not called): P's parameters keep their values and AdamW's
    state for them (no decay, no step); P has no state before its first update."""
    loop = Loop(tmp_path, **EVERY_STEP).write()
    loop.run()
    steps, _ = train(loop.world, loop.config)
    first = next(s.t for s in steps if s.losses["P"] is not None)
    idle = [s.t for s in steps if s.losses["P"] is None and s.t > first]
    assert idle
    pref = indices("pref")
    for t in range(1, first):
        assert not set(pref) & set(loop.checkpoint(t - 1)["optimizer"]["state"])
    for t in idle:
        before, after = loop.checkpoint(t - 2), loop.checkpoint(t - 1)
        assert all(a.equal(b) for a, b in zip(p_tensors(after), p_tensors(before)))
        for i in pref:
            sa, sb = after["optimizer"]["state"][i], before["optimizer"]["state"][i]
            assert all(sa[k].equal(sb[k]) for k in ("step", "exp_avg", "exp_avg_sq"))


def test_parameter_with_zero_gradient_decayed(tmp_path):
    """k of the loss of pi (gradient 0, not None): k_t = k_0 prod_s (1 - eta_s lambda) and
    m = v = 0 with n = t after every step t."""
    loop = Loop(tmp_path, **EVERY_STEP).write()
    loop.run()
    k = fakes()["pi_loss"].k.detach().clone()
    decay = loop.config["groups"]["idle"]["weight_decay"]
    (i,) = indices("idle")
    for t in range(1, 11):
        (entry,) = loop.losses(t - 1)
        k = k * (1 - entry["lr"]["idle"] * decay)
        ck = loop.checkpoint(t - 1)
        torch.testing.assert_close(ck["modules"]["pi_loss"]["k"], k, rtol=1e-12, atol=0)
        state = ck["optimizer"]["state"][i]
        assert float(state["step"]) == t
        assert not state["exp_avg"].any() and not state["exp_avg_sq"].any()
