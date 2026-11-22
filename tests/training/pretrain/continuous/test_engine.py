"""Continuous loss scheduling, normalization and stop-gradient contract oracles."""

import copy

import torch
from torch.nn import functional as F
from training.pretrain.continuous.engine_world import (
    Index,
    config,
    context,
    endpoint,
    observation_step,
    observations,
)

from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.state import Stream


class Capture:
    def __init__(self):
        self.rows = []

    def before_update(self, ctx, result):
        self.rows.append(
            {
                name: [
                    None if p.grad is None else p.grad.clone()
                    for p in module.parameters()
                ]
                for name, module in ctx.modules.items()
            }
        )


def test_k_one_ce_trains_phi_delayed_shift_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg = config(w_phi=0.0)
    ctx = context(tmp_path, cfg)
    original = copy.deepcopy(ctx.modules)
    rows = observations("x", [1, 2], actions=[["b"]])
    pop = {"N_phi": 1, "N_pi": 1, "N_P": 0, "M": 2}
    archive, capture = Archive(tmp_path / "archive" / "shards", 1, 1), Capture()
    stream, result = steps.window(ctx, Stream(), archive, rows[:1], pop, capture)
    z = original["phi"].weight[None, None]
    h = original["pi"].readout(z, torch.tensor([[[0.0, 0.0, 1.0]]]))[:, -1]
    expected = (
        2 * cfg.w_pi * F.cross_entropy(original["pi"].logits(h), torch.tensor([1]))
    )
    expected.backward()
    torch.testing.assert_close(capture.rows[0]["phi"][0], original["phi"].weight.grad)
    assert capture.rows[0]["phi"][0].abs().sum() > 0
    assert result.counts == {"phi": 0, "ce": 1, "shift": 0, "P": 0}
    ctx.step += 1
    stream, result = steps.window(ctx, stream, archive, rows[1:2], pop, capture)
    assert result.counts == {"phi": 1, "ce": 0, "shift": 1, "P": 0}
    assert capture.rows[1]["phi"][0].abs().sum() == 0
    assert any(p is not None and p.abs().sum() > 0 for p in capture.rows[1]["pi"])
    assert stream.pending is None and stream.recurrent.grad_fn is None


def test_preference_older_endpoint_detached_and_original_orientation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg = config(w_phi=0.0, w_pi=0.0, tbptt_states=2)
    edge = {"id": "edge", "a": "old", "b": "new", "labels": {"inc": 0, "p": 0.8}}
    index = Index({}, [edge], {})
    ctx, capture = context(tmp_path, cfg, index=index), Capture()
    original = copy.deepcopy(ctx.modules)
    rows = observations(
        "x",
        [1, 3],
        actions=[[]],
        endpoints={
            0: [endpoint("old", ["x0"], 1)],
            1: [endpoint("new", ["x1"], 1)],
        },
    )
    archive = Archive(tmp_path / "archive" / "shards", 1, 1)
    _, result = steps.window(
        ctx, Stream(), archive, rows, {"N_phi": 0, "N_pi": 0, "N_P": 1, "M": 1}, capture
    )
    w = original["phi"].weight
    old, new = w.detach()[None, None], (3.5 * w)[None, None]
    s, t = original["P"](old, new)
    expected = cfg.w_P * (
        2 * F.binary_cross_entropy_with_logits(t, torch.zeros_like(t))
        + 2 * F.binary_cross_entropy_with_logits(s, torch.full_like(s, 0.8))
    )
    expected.backward()
    torch.testing.assert_close(capture.rows[0]["phi"][0], w.grad)
    torch.testing.assert_close(result.weighted["P"], expected.detach())
    assert (
        result.counts["P"] == 1
        and set(archive.consumed) == {"edge"}
        and not archive.live
    )


def test_empty_window_does_not_decay_or_initialize_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    ctx = context(tmp_path)
    before = copy.deepcopy(ctx.modules["phi"].state_dict())
    archive = Archive(tmp_path / "archive" / "shards", 1, 1)
    stream, result = steps.window(
        ctx,
        Stream(),
        archive,
        observations("x", [1]),
        {"N_phi": 0, "N_pi": 0, "N_P": 0, "M": 1},
    )
    assert not ctx.optimizer.state
    assert result.record(1)["loss"] == 0 and result.lr
    torch.testing.assert_close(ctx.modules["phi"].weight, before["weight"])
    assert stream.recurrent is not None


def test_idle_history_kept_and_unknown_history_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    for prefix, first, expected in (
        ("idle", [], ["idle0", "idle1"]),
        ("gap", ["?"], ["gap1"]),
    ):
        ctx = context(tmp_path)
        rows = observations(prefix, [1, 2, 3, 4], actions=[first, ["a"], ["b"]])
        stream, _ = steps.window(
            ctx,
            Stream(),
            Archive(tmp_path / prefix / "shards", 1, 1),
            rows[:2],
            {"N_phi": 2, "N_pi": 2, "N_P": 0, "M": 2},
        )
        assert stream.pending["indices"] == expected
        torch.testing.assert_close(
            stream.pending["C"][0, -1], torch.tensor([0.0, 0.0, 1.0])
        )
        if not first:
            torch.testing.assert_close(
                stream.pending["C"][0, 0], torch.tensor([0.0, 0.0, 1.0])
            )
