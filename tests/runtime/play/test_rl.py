"""Two real RL cycles with isolated effective configs and tiny CPU networks."""

import json

import pytest
import torch
import yaml
from runtime.play.conftest import PLAY
from training.pretrain.classical.data.index_run import repo_root
from training.rl.cycle_world import Env, install, prepare

from mcpref.config import read_yaml
from mcpref.runtime.play import run


@pytest.mark.parametrize("intermediate", [False, True])
def test_two_real_cycles_keep_world_model_and_optimizer_continuity(
    tmp_path, repo, monkeypatch, intermediate
):
    root, cycle_path = prepare(tmp_path, repo)
    target = root / "configs/training/rl/cycle.yaml"
    target.write_text(cycle_path.read_text())
    path = root / "configs/runtime/play.yaml"
    path.write_text(yaml.safe_dump(PLAY | {"mode": "rl"}))
    checkpoint = None
    if intermediate:
        pretrain = root / "data/pretrain-scaled-init"
        checkpoint = root / "data/live/checkpoints/000007/checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        weights = torch.load(
            pretrain / "model.pt", map_location="cpu", weights_only=True
        )
        torch.save({"modules": weights, "optimizer": {"not": "restored"}}, checkpoint)
        (pretrain / "model.pt").unlink()
        (pretrain / "run.json").write_text('{"complete": false}')
        sampling = root / PLAY["sampling"]
        sampling.write_text(
            yaml.safe_dump(
                read_yaml(sampling) | {"checkpoint_dir": "data/live/checkpoints/000007"}
            )
        )
    events, batches, optimizer_paths = [], [], []
    install(monkeypatch, events, batches)
    env = Env(events)
    env.close = lambda: events.append(("close", env.index))

    def create_env(cfg):
        if checkpoint is not None:
            checkpoint.unlink()
        return env

    monkeypatch.setattr(run, "Env", create_env)
    from mcpref.training.rl import cycle

    original_train = cycle.train

    def train(cfg, modules, index, optimizer):
        optimizer_paths.append(optimizer)
        return original_train(cfg, modules, index, optimizer)

    monkeypatch.setattr(cycle, "train", train)
    with repo_root(root):
        out = run.run(path)
    assert events[-1] == ("close", 4)
    assert [event for event in events if event[0] == "sampler-reset"] == [
        ("sampler-reset", 0),
        ("sampler-reset", 2),
    ]
    assert optimizer_paths[0] is None
    assert optimizer_paths[1].is_relative_to(out / "rl/000000/checkpoints")
    previous = torch.load(optimizer_paths[1], map_location="cpu", weights_only=True)
    assert previous["optimizer"]["state"]
    assert json.loads((out / "run.json").read_text())["completed_trajectories"] == 2
    for index in range(2):
        assert json.loads((out / f"rl/{index:06d}/run.json").read_text())["complete"]
        assert (out / f"sampling/shards/{index:06d}").is_dir()
        assert (out / f"pairs/{index:06d}").is_dir()
    assert (out / "history/000001").is_dir()
    assert (out / "diagnostics").is_dir()
    assert not any(
        (root / "data" / name).exists()
        for name in (
            "sampling",
            "rl_cycle",
            "rl_preference_pairs",
            "rl_preference_history",
            "diagnostics",
        )
    )
