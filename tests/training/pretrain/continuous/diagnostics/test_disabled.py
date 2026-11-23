"""Disabled diagnostic paths must not execute or change training/recovery."""

import json

import pytest
import torch
from training.pretrain.continuous.diagnostics.hooks_world import (
    equal_weights,
    setup,
    training,
    weights,
)
from training.pretrain.continuous.diagnostics.test_hooks import InterruptedDiagnostics

from mcpref.training.pretrain.continuous import loop
from mcpref.training.pretrain.continuous.diagnostics import hooks


def forbidden(*args, **kwargs):
    raise AssertionError("disabled diagnostics executed")


@pytest.mark.parametrize("publish", [False, True])
def test_gradient_diagnostics_false_skips_measurements_and_artifacts(
    tmp_path, monkeypatch, publish
):
    world, data = setup(tmp_path, monkeypatch)
    try:
        ctx = training(tmp_path, world, data, "disabled")
        ctx.cfg.publish_diagnostics = publish
        for name in ("Gradients", "gradient_stats", "update_stats"):
            monkeypatch.setattr(hooks, name, forbidden)
        loop.train(ctx, hooks.Diagnostics())
        assert not (ctx.root / "gradient-flow").exists()
        assert not list(ctx.root.glob("checkpoints/*/gradient_flow"))
        for path in ctx.root.glob("checkpoints/*/losses.json"):
            for row in json.loads(path.read_text()):
                assert row["gradients"] is None and row["updates"] is None
    finally:
        data.close()


@pytest.mark.parametrize("gradients", [False, True])
def test_publisher_disabled_preserves_local_results_and_resume(
    tmp_path, monkeypatch, gradients
):
    world, data = setup(tmp_path, monkeypatch)
    try:
        full = training(tmp_path, world, data, "full", gradients=gradients)
        full.cfg.publish_diagnostics = True
        torch.manual_seed(31)
        loop.train(full, hooks.Diagnostics())
        rng = torch.get_rng_state()
        monkeypatch.setattr(hooks.DiagnosticRun, "load", forbidden)
        stopped = training(tmp_path, world, data, "resumed", gradients=gradients)
        stopped.cfg.publish_diagnostics = False
        torch.manual_seed(31)
        with pytest.raises(InterruptedError, match="checkpoint committed"):
            loop.train(stopped, InterruptedDiagnostics())
        resumed = training(tmp_path, world, data, "resumed", gradients=gradients)
        resumed.cfg.publish_diagnostics = False
        loop.train(resumed, hooks.Diagnostics())
        assert not (tmp_path / "resumed-diagnostics").exists()
        equal_weights(weights(full.root), weights(resumed.root))
        assert torch.equal(torch.get_rng_state(), rng)
        for pattern in (
            "checkpoints/*/losses.json",
            "checkpoints/*/diagnostics.json",
            "reference/*/metrics.json",
            "reference/*/pairs.json",
        ):
            paths = list(full.root.glob(pattern))
            assert paths
            for path in paths:
                other = resumed.root / path.relative_to(full.root)
                assert json.loads(path.read_text()) == json.loads(other.read_text())
    finally:
        data.close()
