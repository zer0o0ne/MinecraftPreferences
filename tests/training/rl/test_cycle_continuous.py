"""Continuous world memory, bounded replay, and CPU checkpoint recovery end to end."""

import json

import pytest
import torch
import tqdm

from training.rl.continuous_cycle_world import equal_state, saved, world
from training.rl.cycle_world import read, run
from training.rl.pairs.helpers import work


@pytest.mark.parametrize("retrain", [False, True])
def test_two_cycles_keep_live_memory_and_train_archived_branches(
    tmp_path, repo, monkeypatch, retrain
):
    root, path, env, live = world(tmp_path, repo, monkeypatch, retrain=retrain)
    out0 = run(root, path, env)
    model0 = saved(out0 / "model.pt")
    state0 = saved(root / "data/sampling/shards/000000/state.pt")
    out1 = run(root, path, env)
    state1 = saved(root / "data/sampling/shards/000001/state.pt")
    assert [index for index, _ in live] == list(range(5))
    assert [scale for _, scale in live[:3]] == [1.2] * 3
    assert [scale for _, scale in live[3:]] == [model0["phi"]["scale"].item()] * 2
    assert model0["phi"]["scale"].item() != 1.2
    assert env.resets == env.restores == 1
    memory = state0["stream"]["layers"][0]["recurrent"]
    for index in (3, 4):
        memory = 0.9 * memory + model0["phi"]["scale"] * (index + 1)
    assert state1["stream"]["layers"][0]["recurrent"].equal(memory)
    assert state1["stream"]["origin"] == {"id": 0}
    assert state1["history"]["Y"].shape[1] == 5
    assert read(out1 / "index/run.json")["counts"]["pending"] == 1
    pairs = work(root / "data/rl_preference_pairs/000001")
    assert any(
        ref["episode"].endswith("#000000")
        for pair in pairs
        for comparison in pair["comparisons"]
        for ref in (comparison["a"], comparison["b"])
    )
    for out in (out0, out1):
        assert read(out / "run.json")["complete"]
        assert read(out / "run.json")["sha256"]["anchors_manifest"]
        assert saved(out / "model.pt")["pretrain_mode"] == "continuous"
        if retrain:
            assert read(out / "retrain/run.json")["complete"]
            rows = [
                row
                for file in (out / "retrain/advantages/shards").glob("*/items.json")
                for row in json.loads(file.read_text())
            ]
            assert len(rows) == read(out / "index/run.json")["counts"]["ppo"]


@pytest.mark.parametrize("phase", ["rl cycle", "rl retrain"])
def test_continuous_rl_resume_matches_uninterrupted_cpu(
    tmp_path, repo, monkeypatch, phase
):
    whole, path, env, _ = world(tmp_path / "whole", repo, monkeypatch, retrain=True)
    run(whole, path, env)
    expected = run(whole, path, env)
    cut, path, env, _ = world(tmp_path / "cut", repo, monkeypatch, retrain=True)
    run(cut, path, env)
    update = tqdm.tqdm.update

    def interrupted(self, n=1):
        result = update(self, n)
        if self.desc == phase and self.n == 1:
            raise RuntimeError("test interruption")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(tqdm.tqdm, "update", interrupted)
        with pytest.raises(RuntimeError, match="test interruption"):
            run(cut, path, env)
    torch.manual_seed(181)
    torch.rand(7)
    actual = run(cut, path, env)
    equal_state(saved(actual / "model.pt"), saved(expected / "model.pt"))
    for stage in ("checkpoints", "retrain/checkpoints"):
        a = saved(max((actual / stage).glob("*/checkpoint.pt")))
        b = saved(max((expected / stage).glob("*/checkpoint.pt")))
        equal_state(a, b)
    equal_state(
        saved(cut / "data/sampling/shards/000001/state.pt"),
        saved(whole / "data/sampling/shards/000001/state.pt"),
    )


def test_changed_historical_anchors_fail_before_sampling_or_training(
    tmp_path, repo, monkeypatch
):
    root, path, env, _ = world(tmp_path, repo, monkeypatch, retrain=False)
    run(root, path, env)
    manifest = root / "data/sampling/shards/000000/anchors.json"
    manifest.write_text(manifest.read_text() + "\n")
    monkeypatch.setattr(
        "mcpref.training.rl.cycle.train", lambda *args: pytest.fail("training")
    )
    with pytest.raises(RuntimeError, match="historical memory anchors"):
        run(root, path, env)
    assert env.index == 2 and env.restores == 0
