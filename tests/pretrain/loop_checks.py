"""Comparisons of the outputs of `pretrain` (specs/pretrain.md Interface: checkpoints,
`losses.json`, `model.pt`) with the steps of the oracle (loop_oracle.py)."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from .loop_modules import fakes, group_order
from .loop_oracle import Step
from .loop_run import Loop

KEYS = {"step", "epoch", "schedule_rng", "batch_in_epoch", "modules", "optimizer",
        "torch_rng", "cuda_rng"}  # fmt: skip


def checkpoint_steps(T: int, n_c: int) -> list[int]:
    """(T) 3: the steps t of the checkpoints, N_c | t or t = T."""
    return [t for t in range(1, T + 1) if t % n_c == 0 or t == T]


def assert_entry(got: dict, want: dict) -> None:
    """An entry of losses.json: its keys in order, the step, the losses (7), lr (5)."""
    assert list(got) == ["step", "phi", "pi", "P", "loss", "lr"]
    assert got["step"] == want["step"] and type(got["step"]) is int
    for k in ("phi", "pi", "loss"):
        assert got[k] == pytest.approx(want[k], rel=1e-9, abs=1e-12), k
    assert (got["P"] is None) == (want["P"] is None)
    if want["P"] is not None:
        assert got["P"] == pytest.approx(want["P"], rel=1e-9)
    assert list(got["lr"]) == list(want["lr"])
    assert got["lr"] == pytest.approx(want["lr"], rel=1e-12)


def assert_states(got: dict, want: dict) -> None:
    """{module name: state_dict}: the same names and entries, CPU tensors close to want."""
    assert set(got) == set(want)
    for k, entries in want.items():
        assert set(got[k]) == set(entries)
        for n, v in entries.items():
            assert (
                got[k][n].device == torch.device("cpu") and not got[k][n].requires_grad
            )
            assert_close(got[k][n], v, msg=f"{k}.{n}")


def assert_optimizer(got: dict, step: Step, cfg: dict, groups: list) -> None:
    """AdamW's state_dict after step t: its `groups` in group order with (5) and the
    config; each parameter's n, m, v as (6), none for a parameter never updated."""
    assert len(got["param_groups"]) == len(groups)
    index = 0
    for pg, (q, ps) in zip(got["param_groups"], groups):
        assert pg["params"] == list(range(index, index + len(ps)))
        index += len(ps)
        assert pg["lr"] == pytest.approx(step.losses["lr"][q], rel=1e-12)
        assert pg["weight_decay"] == cfg["groups"][q]["weight_decay"]
        assert (
            tuple(pg["betas"]) == tuple(cfg["betas"]) and pg["eps"] == cfg["adam_eps"]
        )
        assert pg["amsgrad"] is False and pg["maximize"] is False
        assert pg["decoupled_weight_decay"] is True
    assert sorted(got["state"]) == [i for i, st in enumerate(step.adam) if st]
    for i, st in enumerate(step.adam):
        if st:
            assert float(got["state"][i]["step"]) == st.n
            assert_close(got["state"][i]["exp_avg"], torch.from_numpy(np.asarray(st.m)))
            assert_close(
                got["state"][i]["exp_avg_sq"], torch.from_numpy(np.asarray(st.v))
            )


def assert_run(
    loop: Loop, steps: list[Step], states: list, kept: int, make=None
) -> None:
    """Every checkpoint's losses.json, the checkpoint.pt of the last `kept` ones (their
    position, components, AdamW, torch's generator), model.pt; `make()`: the oracle's
    components (default the fakes)."""
    cfg, T = loop.config, len(steps)
    at = checkpoint_steps(T, cfg["checkpoint_steps"])
    groups = group_order(make() if make else fakes(loop.world.n_r > 0))
    for j, t in enumerate(at):
        first = at[j - 1] if j else 0
        got = loop.losses(j)
        assert [e["step"] for e in got] == list(range(first + 1, t + 1))
        for entry, step in zip(got, steps[first:t]):
            assert_entry(entry, step.losses)
        has = (loop.checkpoint_dir(j) / "checkpoint.pt").exists()
        assert has == (j >= len(at) - kept), j
        if has:
            ck, step = loop.checkpoint(j), steps[t - 1]
            assert set(ck) == KEYS
            assert (ck["step"], ck["epoch"], ck["batch_in_epoch"]) == (t, step.epoch, step.batch_in_epoch)  # fmt: skip
            assert ck["schedule_rng"] == states[step.epoch - 1]
            assert_states(ck["modules"], step.modules)
            assert_optimizer(ck["optimizer"], step, cfg, groups)
            assert torch.equal(ck["torch_rng"], step.torch_rng)
            assert ck["cuda_rng"] is None
    assert not (loop.out / "checkpoints" / f"{len(at):06d}").exists()
    assert_states(loop.model(), steps[-1].modules)
