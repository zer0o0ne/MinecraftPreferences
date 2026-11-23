"""Real atomic checkpoints preserve chronological state, archives and RNG."""

import json

import pytest
import torch
from training.pretrain.classical.references.test_reference_resume import nested_equal
from training.pretrain.continuous.engine_world import (
    Index,
    Reader,
    config,
    context,
    endpoint,
    modules,
    observation_step,
    observations,
)

from mcpref.training.pretrain.continuous import checkpoints, loop, steps
from mcpref.training.pretrain.continuous.schedule import epoch_updates


def world(k):
    edges = [
        {"id": "AB", "a": "A", "b": "B", "labels": {"inc": 0, "p": 1.0}},
        {"id": "BC", "a": "B", "b": "C", "labels": {"inc": 1, "p": None}},
    ]
    streams = {
        "one": observations(
            "x",
            [1, 2, 4, 3],
            endpoints={
                1: [endpoint("A", ["x0", "x1"], 1)],
                3: [endpoint("B", ["x2", "x3"], 2)],
            },
        ),
        "two": observations(
            "y",
            [3, 1, 2],
            endpoints={
                2: [endpoint("C", ["y1", "y2"], 1)],
            },
        ),
    }
    population = {
        "N_phi": 5,
        "N_pi": 5,
        "N_P": 2,
        "M": sum((len(s) + k - 1) // k for s in streams.values()),
    }
    return Index(streams, edges, population)


class Stop:
    def __init__(self, step, when):
        self.step, self.when = step, when

    def after_update(self, ctx, result):
        if self.when == "updated" and ctx.step == self.step:
            raise InterruptedError("test interruption after update")

    def checkpoint(self, ctx, staged, results):
        if self.when == "staged" and ctx.step == self.step:
            raise InterruptedError("test interruption inside checkpoint")

    def published(self, ctx, k):
        if self.when == "published" and ctx.step == self.step:
            raise InterruptedError("test interruption after checkpoint commit")


def compare(root, other):
    first = torch.load(root / "model.pt", weights_only=True)
    second = torch.load(other / "model.pt", weights_only=True)
    assert first.keys() == second.keys()
    for name in first:
        if not isinstance(first[name], dict):
            assert first[name] == second[name]
        else:
            for parameter in first[name]:
                torch.testing.assert_close(
                    first[name][parameter], second[name][parameter], rtol=0, atol=0
                )
    paths = sorted(root.glob("checkpoints/*/losses.json"))
    assert paths
    for path in paths:
        assert json.loads(path.read_text()) == json.loads(
            (other / path.relative_to(root)).read_text()
        )
        assert (
            json.loads(path.with_name("metadata.json").read_text())["memory"]
            == "fragment"
        )


@pytest.mark.parametrize(
    "k,stop,when",
    [(1, 3, "updated"), (1, 5, "staged"), (2, 4, "published"), (2, 8, "published")],
)
def test_resume_matches_uninterrupted(tmp_path, monkeypatch, k, stop, when):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg, index = config(tbptt_states=k), world(k)
    total = epoch_updates(index, cfg) * cfg.epochs
    expected = context(tmp_path / "full", cfg, modules(dropout=0.3), index, total)
    expected.root.mkdir()
    torch.manual_seed(39)
    loop.train(expected)
    interrupted = context(tmp_path / "resumed", cfg, modules(dropout=0.3), index, total)
    interrupted.root.mkdir()
    torch.manual_seed(39)
    with pytest.raises(InterruptedError):
        loop.train(interrupted, Stop(stop, when))
    resumed = context(interrupted.root, cfg, modules(dropout=0.3), index, total)
    torch.manual_seed(987)
    loop.train(resumed)
    compare(expected.root, resumed.root)


def test_retrain_replays_fragments_with_frozen_phi(tmp_path, monkeypatch):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg, index, model = (
        config(epochs=1, tbptt_states=2, batch_size=2),
        world(2),
        modules(),
    )
    loop.phase(cfg, model, index, tmp_path / "joint", "joint", None)
    before = {
        key: value.detach().clone() for key, value in model["phi"].state_dict().items()
    }
    head = {
        key: value.detach().clone() for key, value in model["pi"].state_dict().items()
    }
    loop.phase(cfg, model, index, tmp_path / "retrain", "retrain", None)
    for key, value in model["phi"].state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(
        not torch.equal(value, head[key])
        for key, value in model["pi"].state_dict().items()
    )
    saved = torch.load(
        tmp_path / "retrain/checkpoints/000003/checkpoint.pt", weights_only=False
    )
    assert len(saved["optimizer"]["param_groups"]) == 2
    assert saved["position"]["step"] == 4


def test_gc_preserves_model_optimizer_rng_and_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg, index = config(tbptt_states=1, archive_keep_shards=1), world(1)
    total = epoch_updates(index, cfg) * cfg.epochs
    for name in ("collected", "uncollected"):
        ctx = context(tmp_path / name, cfg, modules(dropout=0.3), index, total)
        ctx.root.mkdir()
        torch.manual_seed(39)
        with monkeypatch.context() as patch:
            if name == "uncollected":
                patch.setattr(checkpoints, "collect", lambda *args: None)
            loop.train(ctx)
    collected, uncollected = tmp_path / "collected", tmp_path / "uncollected"
    compare(collected, uncollected)
    for path in collected.glob("checkpoints/*/checkpoint.pt"):
        a = torch.load(path, weights_only=False)
        b = torch.load(uncollected / path.relative_to(collected), weights_only=False)
        for key in ("modules", "optimizer", "position", "torch_rng", "cuda_rng"):
            assert nested_equal(a[key], b[key]), key
    pattern = "archive/epoch-*/shards/*/histories.pt"
    assert len(list(collected.glob(pattern))) < len(list(uncollected.glob(pattern)))
