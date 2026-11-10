"""Frozen RL retraining with fixed advantages and continuous joint Adam state."""

import json

import torch

from mcpref.infrastructure.runs.files import write_json
from mcpref.training.rl.files import initial_optimizer
from mcpref.training.rl.learning.loop import train
from mcpref.training.rl.learning.retrain import retrain
from mcpref.training.rl.pairs.files import write_json_shard
from training.pretrain.data.index_run import repo_root
from training.rl.learning.loop_world import install_steps, write_world
from training.rl.learning.loss_fakes import modules


def run_joint(root, cfg, index, mods):
    with repo_root(root):
        return train(cfg, mods, index, None)


def install_retrain(monkeypatch, calls):
    def prepared(cfg, mods, index_dir):
        rows = [
            {"kind": "fresh", "index": 0, "advantage": 1.25},
            {"kind": "pending", "index": 2, "advantage": -0.5},
        ]
        write_json_shard(
            index_dir.parent / "retrain/advantages/shards", 0, "items.json", rows[:1]
        )
        write_json_shard(
            index_dir.parent / "retrain/advantages/shards", 1, "items.json", rows[1:]
        )
        write_json(index_dir.parent / "retrain/advantages/run.json", {"complete": True})

    def fake(items, mods, **kwargs):
        fixed = kwargs["fixed_advantages"]
        calls.append((items, dict(fixed)))
        draw = torch.rand((), dtype=torch.float64)
        for name in ("pi", "P"):
            for parameter in mods[name].parameters():
                parameter.grad = torch.full_like(parameter, draw)
        value = float(draw)
        return {"ppo": value, "shift": value + 1, "metra": None,
                "P": value + 2, "loss": value + 3}  # fmt: skip

    monkeypatch.setattr(
        "mcpref.training.rl.learning.retrain.prepare_advantages", prepared
    )
    monkeypatch.setattr("mcpref.training.rl.learning.retrain.step_gradients", fake)


def run_retrain(root, cfg, index, mods, optimizer):
    with repo_root(root):
        retrain(cfg, mods, index, optimizer)


def final_checkpoint(index):
    paths = sorted((index.parent / "retrain/checkpoints").glob("*/checkpoint.pt"))
    return torch.load(paths[-1], map_location="cpu", weights_only=True)


def assert_models_equal(a, b):
    assert a.keys() == b.keys()
    for name in a:
        for key in a[name]:
            assert a[name][key].equal(b[name][key])


def test_retrain_uses_fixed_advantages_freezes_phi_and_continues_adam(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_world(tmp_path, repo, retrain=True)
    mods = modules()
    install_steps(monkeypatch, [])
    optimizer = run_joint(root, cfg, index, mods)
    joint_phi = mods["phi"].scale.detach().clone()
    joint_dual = mods["metra"].log_lambda.detach().clone()
    joint_state = optimizer.state_dict()["state"]
    calls = []
    install_retrain(monkeypatch, calls)
    monkeypatch.setattr(
        "mcpref.training.rl.learning.retrain.adamw",
        lambda *args: (_ for _ in ()).throw(AssertionError("new optimizer created")),
    )
    run_retrain(root, cfg, index, mods, optimizer)
    assert mods["phi"].scale.detach().equal(joint_phi)
    assert mods["metra"].log_lambda.detach().equal(joint_dual)
    assert len(calls) == 6
    assert all(fixed for _, fixed in calls)
    assert [fixed for _, fixed in calls if ("fresh", 0) in fixed] == [
        {("fresh", 0): 1.25}
    ] * 4
    assert [fixed for _, fixed in calls if ("pending", 2) in fixed] == [
        {("pending", 2): -0.5}
    ] * 2
    saved = final_checkpoint(index)
    assert saved["step"] == 6 and set(saved["optimizer"]["state"]) >= set(joint_state)
    assert json.loads((index.parent / "retrain/run.json").read_text())["complete"]
    with repo_root(root):
        source = initial_optimizer(cfg, 1)
    assert source.parent.parent.name == "checkpoints"
    assert source.parent.parent.parent.name == "retrain"


def test_interrupted_retrain_restores_same_joint_optimizer_and_modules(
    tmp_path, repo, monkeypatch
):
    whole, cfg, whole_index = write_world(tmp_path / "whole", repo, retrain=True)
    mods = modules()
    install_steps(monkeypatch, [])
    optimizer = run_joint(whole, cfg, whole_index, mods)
    install_retrain(monkeypatch, [])
    run_retrain(whole, cfg, whole_index, mods, optimizer)
    expected_model = torch.load(
        whole_index.parent / "model.pt", map_location="cpu", weights_only=True
    )
    expected_optimizer = final_checkpoint(whole_index)["optimizer"]

    cut, cfg, cut_index = write_world(tmp_path / "cut", repo, retrain=True)
    mods = modules()
    install_steps(monkeypatch, [])
    optimizer = run_joint(cut, cfg, cut_index, mods)
    install_retrain(monkeypatch, [])
    real_update = __import__("tqdm").tqdm.update
    raised = False

    def interrupt(self, n=1):
        nonlocal raised
        result = real_update(self, n)
        if self.desc == "rl retrain" and self.n == 2 and not raised:
            raised = True
            raise RuntimeError("interrupt retrain")
        return result

    monkeypatch.setattr(__import__("tqdm").tqdm, "update", interrupt)
    try:
        run_retrain(cut, cfg, cut_index, mods, optimizer)
    except RuntimeError as error:
        assert str(error) == "interrupt retrain"
    monkeypatch.setattr(__import__("tqdm").tqdm, "update", real_update)
    run_retrain(cut, cfg, cut_index, modules(), None)
    actual_model = torch.load(
        cut_index.parent / "model.pt", map_location="cpu", weights_only=True
    )
    assert_models_equal(actual_model, expected_model)
    actual_optimizer = final_checkpoint(cut_index)["optimizer"]
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key in expected_optimizer["state"]:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert actual_optimizer["state"][key][field].equal(
                expected_optimizer["state"][key][field]
            )


def test_advantage_shards_are_computed_once_from_joint_preference(
    tmp_path, repo, monkeypatch
):
    from mcpref.training.rl.learning.retrain import prepare_advantages
    from mcpref.training.rl.pairs.files import read_json_shard
    from training.rl.learning.loss_fakes import history

    root, cfg, index = write_world(tmp_path, repo, retrain=True)
    mods = modules()
    calls = []

    def encoded(ref, phi, perception, sampling, states, action_count):
        return history(ref.index, phi)

    def rewarded(preference, origins, futures, horizon, discount, min_p):
        calls.append(preference.weight.detach().clone())
        return preference.weight.sum().reshape(1)

    monkeypatch.setattr("mcpref.training.rl.learning.retrain.encode_ref", encoded)
    monkeypatch.setattr("mcpref.training.rl.learning.retrain.reward", rewarded)
    with repo_root(root):
        prepare_advantages(cfg, mods, index)
    first = read_json_shard(index.parent / "retrain/advantages/shards", 0, "items.json")
    second = read_json_shard(
        index.parent / "retrain/advantages/shards", 1, "items.json"
    )
    expected = mods["P"].weight.sum().item()
    assert first == [{"kind": "fresh", "index": 0, "advantage": expected}]
    assert second == [{"kind": "pending", "index": 2, "advantage": expected}]
    assert len(calls) == 2
    with torch.no_grad():
        mods["P"].weight.fill_(10)
    with repo_root(root):
        prepare_advantages(cfg, mods, index)
    assert len(calls) == 2
    assert (
        read_json_shard(index.parent / "retrain/advantages/shards", 0, "items.json")
        == first
    )


def test_new_process_retrain_restores_final_joint_checkpoint(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_world(tmp_path, repo, retrain=True)
    mods = modules()
    install_steps(monkeypatch, [])
    run_joint(root, cfg, index, mods)
    calls = []
    install_retrain(monkeypatch, calls)
    run_retrain(root, cfg, index, mods, None)
    assert len(calls) == 6
    assert final_checkpoint(index)["step"] == 6
    assert json.loads((index.parent / "retrain/run.json").read_text())["complete"]
