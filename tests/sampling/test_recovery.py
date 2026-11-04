"""Exact continuation, model changes, atomic recovery and run files of env-sampling.md."""

import hashlib
import json
import shutil

import numpy as np
import pytest
import torch
import yaml
from progress_bars import Bar, assert_progress, states

from mcpref.mcts.tree import advance as real_advance
from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import load_state, load_step

from .helpers import FakeEnv, modules


def assert_value(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_value(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_value(a, b)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def assert_record(actual, expected):
    assert actual.elapsed_ms >= 0.0
    for name in (
        "index", "action", "probabilities", "delta", "predicted", "preference_close",
        "preference_start", "tree",
    ):  # fmt: skip
        assert_value(getattr(actual, name), getattr(expected, name))
    for name in ("before", "after"):
        assert_value(getattr(actual, name).frame, getattr(expected, name).frame)
        assert getattr(actual, name).line == getattr(expected, name).line
        np.testing.assert_array_equal(getattr(actual, name).mask, getattr(expected, name).mask)  # fmt: skip


def test_unchanged_model_restore_matches_uninterrupted_second_trajectory(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    before_run = site.out / "run.json"
    saved_run = site.out.parent / "saved-run.json"
    shutil.copy2(before_run, saved_run)
    sampler.sample()
    expected = [load_step(site.out, 2), load_step(site.out, 3)]
    shutil.rmtree(site.out / "shards/000001")
    shutil.copy2(saved_run, before_run)
    env = FakeEnv()
    env.index = 2
    restored = Sampler(site.cfg, env, modules(events=site.events))
    restored.reset()
    assert env.resets == 0
    restored.sample()
    for index, wanted in zip((2, 3), expected, strict=True):
        assert_record(load_step(site.out, index), wanted)


def test_changed_model_rebuilds_suffix_but_keeps_stale_root_statistics(
    site, monkeypatch
):
    from mcts.helpers import node

    from mcpref.mcts.search import Decision
    from mcpref.sampling.shards import write_state

    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    old = load_state(site.out, site.modules.phi, 2, torch.device("cpu"))
    child = node(
        [9.0, 9.0], x=[10.0, 10.0], action=0, prior=0.5,
        shift=[0.0, 0.0], parent=old.tree.root, n=7, mean=0.25, value=0.4,
    )  # fmt: skip
    old.tree.root.children = [child]
    write_state(site.out / "shards/000000", old)
    previous = load_step(site.out, 1)
    events = []
    changed = modules("changed-model", events=events, offset=100.0)
    env = FakeEnv(events)
    env.index = 2
    advances = []

    def advance(tree, action, real, alpha, epsilon):
        advances.append((action, real.Y[0, -1, 0].item()))
        return real_advance(tree, action, real, alpha, epsilon)

    def decide(tree, mask, pi, P, alpha, cfg, rng, deadline):
        if not tree.root.children:
            tree.root.children = [
                node([0.0, 0.0], action=0, prior=1.0, parent=tree.root)
            ]
        return Decision(0, np.array([1.0, 0.0, 0.0]))

    def assert_stale(raw, expected):
        for name in ("z", "x", "shift"):
            actual = raw[name]
            wanted = getattr(expected, name)
            if wanted is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, wanted)
        for name in ("action", "prior", "n", "mean", "value"):
            assert raw[name] == getattr(expected, name)
        assert len(raw["children"]) == len(expected.children)
        for actual_child, expected_child in zip(
            raw["children"], expected.children, strict=True
        ):
            assert_stale(actual_child, expected_child)

    for name in ("mcpref.sampling.steps.advance", "mcpref.sampling.history.advance"):
        monkeypatch.setattr(name, advance, raising=False)
    monkeypatch.setattr("mcpref.sampling.steps.decide", decide)
    restored = Sampler(site.cfg, env, changed)
    restored.reset()
    assert env.resets == 0
    restored.sample()
    current = load_step(site.out, 2)
    assert_stale(current.tree["root"], old.tree.root)
    base = current.tree["base"]
    expected_y = torch.tensor([[[101.0, 101.5], [102.0, 102.5]]], dtype=torch.float64)
    assert base["Y"].tolist() == expected_y.tolist()
    torch.testing.assert_close(base["C"], old.history.C)
    assert base["C"][0, 1, previous.action] == 1.0
    streamed = [event[1] for event in events if event[0] == "stream"]
    assert streamed == [0, 1, 2, 3, 4]
    following = load_step(site.out, 3)
    assert advances == [(current.action, 103.0), (following.action, 104.0)]


def test_partial_trajectory_repeated_complete_shard_untouched_and_progress(
    site, capsys
):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    complete = site.out / "shards/000000"
    before = {p.relative_to(complete): p.read_bytes() for p in complete.rglob("*") if p.is_file()}  # fmt: skip
    partial = site.out / "shards/000001.part/steps"
    partial.mkdir(parents=True)
    (partial / "000000.pt").write_bytes(b"interrupted")
    env = FakeEnv()
    env.index = 2
    restored = Sampler(site.cfg, env, modules())
    restored.reset()
    capsys.readouterr()
    restored.sample()
    captured = capsys.readouterr()
    assert captured.out == ""
    drawn = states(captured.err, "env sampling")
    assert drawn[0] == Bar("env sampling", 0, 2, "step")
    assert drawn[-1] == Bar("env sampling", 2, 2, "step")
    assert_progress(captured.err)
    assert not (site.out / "shards/000001.part").exists()
    after = {p.relative_to(complete): p.read_bytes() for p in complete.rglob("*") if p.is_file()}  # fmt: skip
    assert after == before


def test_counts_and_models_repaired_after_crash_before_run_rewrite(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    path = site.out / "run.json"
    run = json.loads(path.read_text())
    assert run["counts"] == {"trajectories": 1, "steps": 2}
    assert run["models"] == {"000000": "fake-model"}
    run["counts"], run["models"] = {"trajectories": 0, "steps": 0}, {}
    path.write_text(json.dumps(run))
    Sampler(site.cfg, FakeEnv(), modules())
    repaired = json.loads(path.read_text())
    assert repaired["counts"] == {"trajectories": 1, "steps": 2}
    assert repaired["models"] == {"000000": "fake-model"}


def test_every_ordered_fingerprint_and_static_recheck(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    run = json.loads((site.out / "run.json").read_text())
    order = [
        "config", "pretrain_config", "pretrain_model", "env", "mcts",
        "perception_data", "perception", "action_mask", "policy", "preference",
        "engine", "classifier", "clips", "vpt_actions", "gui_screens",
    ]  # fmt: skip
    assert list(run["sha256"]) == order
    pretrain = yaml.safe_load((site.root / "configs/pretrain.yaml").read_text())
    env = yaml.safe_load((site.root / site.cfg.env).read_text())
    clips = yaml.safe_load((site.root / env["clips"]).read_text())
    masks = yaml.safe_load((site.root / pretrain["action_mask"]).read_text())
    paths = {
        "pretrain_config": site.cfg.pretrain,
        "pretrain_model": f"{pretrain['out_dir']}/model.pt",
        "env": site.cfg.env,
        "mcts": site.cfg.mcts,
        "perception_data": site.cfg.perception_data,
        "perception": pretrain["perception"],
        "action_mask": pretrain["action_mask"],
        "policy": pretrain["policy"],
        "preference": pretrain["preference"],
        "engine": env["engine"],
        "classifier": env["classifier"],
        "clips": env["clips"],
        "vpt_actions": clips["actions"],
        "gui_screens": masks["screens"],
    }
    for name, relative in paths.items():
        digest = hashlib.sha256((site.root / relative).read_bytes()).hexdigest()
        assert run["sha256"][name] == digest
    assert run["sha256"]["config"] == hashlib.sha256(site.path.read_bytes()).hexdigest()
    with (site.root / site.cfg.mcts).open("a") as file:
        file.write("# changed\n")
    with pytest.raises(RuntimeError, match="mcts changed"):
        Sampler(site.cfg, FakeEnv(), modules())
