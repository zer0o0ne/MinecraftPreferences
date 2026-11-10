"""Equation (4), C4: cross-cycle AdamW continuity, compatibility, and exact resume."""

import copy
import json

import numpy as np
import pytest
import torch

from training.pretrain.loop_oracle import Moments, adamw
from training.rl.learning.helpers import GROUPS
from training.rl.learning.loop_world import (
    checkpoints,
    install_steps,
    losses,
    run_train,
    write_unit_world,
)
from training.rl.learning.loss_fakes import modules
from training.rl.learning.test_loop import Interrupted, interrupt_at

GRADIENT = 0.375


def install_constant_gradient(monkeypatch, value=GRADIENT):
    def fake(items, mods, **kwargs):
        mods["phi"].scale.grad = torch.full_like(mods["phi"].scale, value)
        return {"ppo": 0.0, "shift": None, "metra": None, "P": None, "loss": 0.0}

    monkeypatch.setattr("mcpref.training.rl.learning.loop.step_gradients", fake)


def observe_optimizer(monkeypatch):
    from mcpref.training.optim import update

    observed = []

    def wrapped(*args, **kwargs):
        optimizer = next(arg for arg in args if isinstance(arg, torch.optim.Optimizer))
        observed.append(copy.deepcopy(optimizer.state_dict()))
        return update(*args, **kwargs)

    monkeypatch.setattr("mcpref.training.rl.learning.loop.update", wrapped)
    return observed


def previous_cycle(tmp_path, repo, monkeypatch, epochs=1, **changes):
    root, cfg, index = write_unit_world(tmp_path, repo, epochs=epochs, **changes)
    install_constant_gradient(monkeypatch)
    run_train(root, cfg, index, modules())
    return root, index, checkpoints(index)[-1]


def final_optimizer(index):
    return torch.load(checkpoints(index)[-1], map_location="cpu", weights_only=True)[
        "optimizer"
    ]


def assert_adam_bitwise(actual, expected):
    assert actual["state"].keys() == expected["state"].keys()
    for index, state in actual["state"].items():
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value, wanted = state[key], expected["state"][index][key]
            assert value.device.type == "cpu" and wanted.device.type == "cpu"
            assert value.dtype == wanted.dtype and value.equal(wanted)


def assert_final_optimizer_bitwise(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert_adam_bitwise(actual, expected)


def test_cycle_zero_first_update_starts_with_empty_adam_state(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_unit_world(tmp_path, repo)
    install_steps(monkeypatch, [])
    observed = observe_optimizer(monkeypatch)
    run_train(root, cfg, index, modules())
    assert len(observed) == 1 and observed[0]["state"] == {}


def test_cycle_one_preupdate_state_equals_previous_final_cpu_bitwise(
    tmp_path, repo, monkeypatch
):
    root, previous, source = previous_cycle(tmp_path, repo, monkeypatch, epochs=2)
    expected = final_optimizer(previous)
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    run_train(root, cfg, index, modules(), source)
    assert_adam_bitwise(observed[0], expected)
    assert float(observed[0]["state"][0]["step"]) == 2
    assert float(final_optimizer(index)["state"][0]["step"]) == 3


def test_parameters_without_previous_state_remain_unset_before_cycle_one_update(
    tmp_path, repo, monkeypatch
):
    root, previous, source = previous_cycle(tmp_path, repo, monkeypatch)
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    run_train(root, cfg, index, modules(), source)
    assert set(final_optimizer(previous)["state"]) == {0}
    assert set(observed[0]["state"]) == {0}
    assert set(final_optimizer(index)["state"]) == {0}


def test_first_cycle_one_update_matches_persistent_adam_and_differs_from_fresh(
    tmp_path, repo, monkeypatch
):
    root, previous, source = previous_cycle(
        tmp_path, repo, monkeypatch, grad_clip=1.0e9, lr_floor=1.0
    )
    _, cfg, index = write_unit_world(
        tmp_path, repo, cycle=1, grad_clip=1.0e9, lr_floor=1.0
    )
    install_constant_gradient(monkeypatch, 0.625)
    actual = modules()
    expected, fresh = (actual["phi"].scale.detach().clone() for _ in range(2))
    state = final_optimizer(previous)["state"][0]
    moments = Moments(
        int(state["step"]),
        state["exp_avg"].numpy().copy(),
        state["exp_avg_sq"].numpy().copy(),
    )
    options = (cfg.betas, cfg.adam_eps)
    group = cfg.groups["phi_head"]
    gradient = np.array(0.625)
    adamw(expected, gradient, moments, group["lr"], group["weight_decay"], *options)
    adamw(
        fresh,
        gradient,
        Moments(0, np.zeros(()), np.zeros(())),
        group["lr"],
        group["weight_decay"],
        *options,
    )
    run_train(root, cfg, index, actual, source)
    torch.testing.assert_close(actual["phi"].scale, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(actual["phi"].scale, fresh, rtol=1e-12, atol=1e-12)


def test_cycle_one_learning_rate_restarts_at_local_step_one(
    tmp_path, repo, monkeypatch
):
    root, _previous, source = previous_cycle(
        tmp_path, repo, monkeypatch, warmup_steps=4
    )
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1, warmup_steps=4)
    run_train(root, cfg, index, modules(), source)
    entries = [entry for shard in losses(index) for entry in shard]
    assert len(entries) == 1 and entries[0]["step"] == 1
    expected = {name: group["lr"] / 4 for name, group in cfg.groups.items()}
    assert entries[0]["lr"] == pytest.approx(expected, rel=1e-12)


def test_keep_one_retains_final_checkpoint_as_next_cycle_optimizer_source(
    tmp_path, repo, monkeypatch
):
    root, previous, source = previous_cycle(
        tmp_path, repo, monkeypatch, epochs=3, keep_checkpoints=1
    )
    assert [path.parent.name for path in checkpoints(previous)] == ["000002"]
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    run_train(root, cfg, index, modules(), source)
    assert_adam_bitwise(observed[0], final_optimizer(previous))


@pytest.mark.parametrize("resume_source", ["none", "changed-previous"])
def test_current_checkpoint_priority_and_final_optimizer_matches(
    tmp_path, repo, monkeypatch, resume_source
):
    whole, _previous, source = previous_cycle(tmp_path / "whole", repo, monkeypatch)
    _, cfg, whole_index = write_unit_world(tmp_path / "whole", repo, cycle=1, epochs=3)
    run_train(whole, cfg, whole_index, modules(), source)
    cut, _previous, source = previous_cycle(tmp_path / "cut", repo, monkeypatch)
    _, cfg, cut_index = write_unit_world(tmp_path / "cut", repo, cycle=1, epochs=3)
    interrupt_at(monkeypatch, 1)
    with pytest.raises(Interrupted):
        run_train(cut, cfg, cut_index, modules(), source)
    if resume_source == "changed-previous":
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        for state in checkpoint["optimizer"]["state"].values():
            state["exp_avg"].add_(1)
            state["exp_avg_sq"].add_(1)
        torch.save(checkpoint, source)
    else:
        source = None
    run_train(cut, cfg, cut_index, modules(), source)
    assert_final_optimizer_bitwise(
        final_optimizer(cut_index), final_optimizer(whole_index)
    )


def test_cycle_one_without_current_checkpoint_or_previous_source_raises(
    tmp_path, repo, monkeypatch
):
    root, _previous, _source = previous_cycle(tmp_path, repo, monkeypatch)
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    with pytest.raises(RuntimeError):
        run_train(root, cfg, index, modules(), None)
    assert observed == []


def test_missing_previous_checkpoint_raises_before_update(tmp_path, repo, monkeypatch):
    root, _previous, source = previous_cycle(tmp_path, repo, monkeypatch)
    source.unlink()
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    with pytest.raises(FileNotFoundError):
        run_train(root, cfg, index, modules(), source)
    assert observed == []


def test_nonfinal_previous_checkpoint_raises_before_update(tmp_path, repo, monkeypatch):
    root, previous, source = previous_cycle(tmp_path, repo, monkeypatch, epochs=2)
    run_path = previous / "run.json"
    run = json.loads(run_path.read_text())
    run["steps"] = 1
    run_path.write_text(json.dumps(run))
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    with pytest.raises(RuntimeError):
        run_train(root, cfg, index, modules(), source)
    assert observed == []


@pytest.mark.parametrize(
    "changes",
    [
        {"betas": [0.8, 0.9]},
        {"adam_eps": 1.0e-7},
        {"groups": GROUPS | {"phi_vision": {"lr": 1.0e-5, "weight_decay": 0.02}}},
    ],
    ids=["betas", "eps", "weight-decay"],
)
def test_incompatible_previous_adam_hyperparameters_raise_before_update(
    tmp_path, repo, monkeypatch, changes
):
    root, _previous, source = previous_cycle(tmp_path, repo, monkeypatch)
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1, **changes)
    observed = observe_optimizer(monkeypatch)
    with pytest.raises(RuntimeError):
        run_train(root, cfg, index, modules(), source)
    assert observed == []


def test_incompatible_previous_group_parameter_indices_raise_before_update(
    tmp_path, repo, monkeypatch
):
    root, _previous, source = previous_cycle(tmp_path, repo, monkeypatch)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint["optimizer"]["param_groups"][0]["params"] = [0]
    torch.save(checkpoint, source)
    _, cfg, index = write_unit_world(tmp_path, repo, cycle=1)
    observed = observe_optimizer(monkeypatch)
    with pytest.raises(RuntimeError):
        run_train(root, cfg, index, modules(), source)
    assert observed == []
