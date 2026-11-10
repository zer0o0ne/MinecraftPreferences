"""(4)-(6) of specs/pretrain.md through `pretrain`, a checkpoint after every step: each step's
clipping, learning rates and AdamW update against the oracle (loop_oracle.py); the learning
rates of (5) on both branches, $T_w = 0$ and $T_w \\ge T$; a parameter whose gradient is None
untouched, one whose gradient is zero decayed (hand formulas)."""

import dataclasses
import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.training.optim import iter_update_terms
from mcpref.training.pretrain.update import gradient_stats, update_stats
from training.pretrain.loop_checks import assert_run
from training.pretrain.loop_modules import fakes, group_order
from training.pretrain.loop_oracle import train
from training.pretrain.loop_run import Loop

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


class Groups(dict):
    pass


def diagnostic_groups(entries, names) -> Groups:
    groups = Groups(entries)
    groups.parameter_names = {id(p): name for p, name in names.items()}
    return groups


def config(groups):
    from training.pretrain.steps.test_loss import CFG

    return dataclasses.replace(CFG, groups=groups)


def test_preclip_gradient_norms_none_zero_heads_and_clip():
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    zero = torch.nn.Parameter(torch.zeros(2))
    idle = torch.nn.Parameter(torch.ones(1))
    preference = torch.nn.Module()
    preference.pref = torch.nn.Linear(1, 1, bias=False)
    preference.inc = torch.nn.Linear(1, 1)
    parameters = [
        a,
        zero,
        idle,
        preference.pref.weight,
        preference.inc.weight,
        preference.inc.bias,
    ]
    a.grad = torch.tensor([3.0, 4.0])
    zero.grad = torch.zeros(2)
    preference.pref.weight.grad = torch.tensor([[1.0]])
    preference.inc.bias.grad = torch.tensor([2.0])
    groups = diagnostic_groups({"a": [a, zero], "idle": [idle], "P": parameters[3:]},
                               {p: f"m.{i}" for i, p in enumerate(parameters)})  # fmt: skip
    got = gradient_stats(groups, 2.0, preference=preference)
    norm = math.sqrt(30)
    assert got["global"] == {
        "norm": norm,
        "clip_coefficient": 2 / (norm + 1e-6),
        "active": True,
    }
    assert got["groups"]["a"] == {"norm": 5.0, "active": True}
    assert got["groups"]["idle"] == {"norm": None, "active": False}
    assert got["P_heads"] == {"pref_weight": {"norm": 1.0, "active": True},
                              "inc_weight": {"norm": None, "active": False},
                              "inc_bias": {"norm": 2.0, "active": True}}  # fmt: skip
    idle_groups = diagnostic_groups({"idle": [idle]}, {idle: "m.idle"})
    inactive = gradient_stats(idle_groups, 2.0, preference=None)
    assert inactive["global"] == {
        "norm": None,
        "clip_coefficient": None,
        "active": False,
    }
    assert inactive["P_heads"] is None


def adam(groups, cfg, lr, decay):
    return torch.optim.AdamW(
        [
            {"params": ps, "lr": lr[q], "weight_decay": decay[q]}
            for q, ps in groups.items()
        ],
        betas=cfg.betas,
        eps=cfg.adam_eps,
    )


def test_update_terms_and_stats_match_actual_before_after_clones():
    parameters = [
        torch.nn.Parameter(torch.tensor(x)) for x in ([1.0, -2.0], [3.0], [0.0], [4.0])
    ]
    a, b, zero, idle = parameters
    names = {p: f"m.{i}" for i, p in enumerate(parameters)}
    groups = diagnostic_groups({"a": [a, idle], "b": [b, zero]}, names)
    lr, decay = {"a": 0.1, "b": 0.2}, {"a": 0.3, "b": 0.0}
    cfg = config({q: {"lr": lr[q], "weight_decay": decay[q]} for q in lr})
    a.grad, b.grad, zero.grad = (
        torch.tensor([0.5, -0.25]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
    )
    optimizer = adam(groups, cfg, lr, decay)
    before = {names[p]: p.detach().clone() for p in parameters}
    optimizer.step()
    after = {names[p]: p.detach().clone() for p in parameters}
    terms = list(iter_update_terms(cfg, optimizer, groups, lr))
    assert [(x.group, x.parameter) for x in terms] == [
        ("a", "m.0"),
        ("a", "m.3"),
        ("b", "m.1"),
        ("b", "m.2"),
    ]
    for term in terms:
        assert (
            term.unavailable is None
            and term.before is not None
            and term.delta is not None
        )
        assert not term.before.requires_grad and not term.delta.requires_grad
        assert_close(term.before, before[term.parameter])
        assert_close(term.delta, after[term.parameter] - before[term.parameter])
    got = update_stats(cfg, optimizer, groups, lr)["groups"]
    for group, ps in (("a", [a, idle]), ("b", [b, zero])):
        delta = torch.cat(
            [(after[names[p]] - before[names[p]]).reshape(-1) for p in ps]
        )
        previous = torch.cat([before[names[p]].reshape(-1) for p in ps])
        assert got[group]["norm"] == pytest.approx(float(delta.norm()))
        assert got[group]["relative"] == pytest.approx(
            float(delta.norm() / previous.norm())
        )
        assert got[group]["unavailable"] is None
    assert got["b"]["relative"] is not None


def test_update_zero_before_norm_and_zero_decay_factor():
    active = torch.nn.Parameter(torch.zeros(1))
    idle = torch.nn.Parameter(torch.tensor([2.0]))
    parameters = [active, idle]
    groups = diagnostic_groups({"z": parameters}, {active: "m.active", idle: "m.idle"})
    lr, decay = {"z": 1.0}, {"z": 1.0}
    cfg = config({"z": {"lr": 1.0, "weight_decay": 1.0}})
    active.grad = torch.ones(1)
    optimizer = adam(groups, cfg, lr, decay)
    before_idle = idle.detach().clone()
    optimizer.step()
    terms = list(iter_update_terms(cfg, optimizer, groups, lr))
    assert (
        terms[0].before is terms[0].delta is None
        and terms[0].unavailable == "zero_decay_factor"
    )
    assert_close(terms[1].before, before_idle)
    assert_close(terms[1].delta, torch.zeros_like(idle))
    assert terms[1].unavailable is None
    got = update_stats(cfg, optimizer, groups, lr)["groups"]["z"]
    assert got == {"norm": None, "relative": None, "unavailable": "zero_decay_factor"}


def test_update_relative_is_null_for_zero_parameter_norm():
    parameter = torch.nn.Parameter(torch.zeros(1))
    groups = diagnostic_groups({"z": [parameter]}, {parameter: "m.p"})
    lr, decay = {"z": 0.1}, {"z": 0.0}
    cfg = config({"z": {"lr": 0.1, "weight_decay": 0.0}})
    parameter.grad = torch.ones(1)
    optimizer = adam(groups, cfg, lr, decay)
    optimizer.step()
    got = update_stats(cfg, optimizer, groups, lr)["groups"]["z"]
    assert got["norm"] > 0 and got["relative"] is None and got["unavailable"] is None
