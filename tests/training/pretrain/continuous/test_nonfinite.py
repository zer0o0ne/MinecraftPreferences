"""Invalid objectives must stop before any optimizer state is changed."""

import copy

import pytest
import torch
from training.pretrain.continuous.engine_world import (
    context,
    observation_step,
    observations,
)

from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.state import Stream


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_with_finite_gradient_cannot_update(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    monkeypatch.setattr(steps, "reusable", lambda _: None)
    ctx = context(tmp_path)
    before = copy.deepcopy(ctx.modules["phi"].state_dict())
    # A constant invalid offset is invisible to gradient-only finiteness checks.
    monkeypatch.setattr(steps, "own_losses", lambda *args: {"phi": args[3].sum() + bad})
    archive = Archive(tmp_path / "archive/shards", 1, 1)
    try:
        with pytest.raises(FloatingPointError, match="Non-finite continuous loss"):
            steps.window(
                ctx,
                Stream(),
                archive,
                observations("x", [1]),
                {"N_phi": 1, "N_pi": 0, "N_P": 0, "updates": 1},
            )
        assert not ctx.optimizer.state
        for name, value in ctx.modules["phi"].state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    finally:
        archive.close()
