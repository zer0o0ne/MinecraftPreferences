"""Global auxiliary weighting, K=1 gradients and frozen-phi training."""

import copy

import pytest
import torch
from training.pretrain.continuous.engine_world import (
    config,
    context,
    observation_step,
    observations,
)
from training.pretrain.continuous.test_engine import Capture

from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.diagnostics.observations import OwnSummary
from mcpref.training.pretrain.continuous.state import Stream


@pytest.mark.parametrize("frozen", [False, True])
def test_k_one_auxiliary_trains_current_phi_and_heads_independently_of_w_pi(
    tmp_path, monkeypatch, frozen
):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    cfg = config(
        w_phi=0.0, w_pi=0.0, w_change=1.7, calibration_bins=3, action_copy_eps=0.1
    )
    ctx, capture = context(tmp_path, cfg), Capture()
    if frozen:
        ctx.modules["phi"].requires_grad_(False)
    rows = observations("x", [1, 2, 3], actions=[["a"], ["b", "b"]])
    population = {"N_phi": 2, "N_pi": 2, "N_P": 0, "updates": 3}
    archive = Archive(tmp_path / "archive/shards", 1, 1)
    stream, _ = steps.window(ctx, Stream(), archive, rows[:1], population, capture)
    assert capture.rows[0]["pi"][-1] is None
    assert all(not row["z"].requires_grad for row in stream.policy)
    assert not stream.recurrent.requires_grad
    oracle = copy.deepcopy(ctx.modules)
    phi, pi = oracle["phi"], oracle["pi"]
    z = 2 * phi.weight + 0.5 * stream.recurrent
    Y = torch.stack([stream.policy[0]["z"], z])[None]
    C = z.new_tensor([[[0, 0, 1], [1, 0, 0]]])
    readout = pi.readout(Y, C)[:, -1]
    loss = (1 - pi.change(readout)[0]) * -pi.logits(readout).log_softmax(-1)[0, 1]
    expected = 3 * 1.7 / 2 * loss
    expected.backward()
    ctx.step += 1
    _, result = steps.window(ctx, stream, archive, rows[1:2], population, capture)
    for name in ("phi", "pi"):
        for actual, parameter in zip(capture.rows[1][name], oracle[name].parameters()):
            if parameter.grad is None:
                assert actual is None or not actual.any()
            else:
                torch.testing.assert_close(actual, parameter.grad)
    torch.testing.assert_close(result.weighted["pi"], expected.detach())
    torch.testing.assert_close(result.sums["pi_change"], loss.detach())
    assert capture.rows[1]["pi"][-1].abs().sum() > 0
    if not frozen:
        assert capture.rows[1]["phi"][0].abs().sum() > 0
    assert set(result.weighted) == {"phi", "pi"}
    torch.testing.assert_close(
        torch.tensor(result.record(ctx.step)["loss"]), expected.detach()
    )
    summary = OwnSummary(cfg)
    summary.add(result)
    losses = summary.finish(ctx.modules)["losses"]
    assert losses["pi_change"] == float(loss.detach())
    assert losses["pi"] == losses["pi_action"] + losses["pi_shift"]


def test_zero_change_weight_leaves_auxiliary_parameters_without_optimizer_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    ctx = context(tmp_path, config(w_change=0.0))
    change = ctx.modules["pi"].change_head
    before = copy.deepcopy(change.state_dict())
    rows = observations("x", [1, 2, 3], actions=[["a"], ["b"]])
    steps.window(
        ctx,
        Stream(),
        Archive(tmp_path / "archive/shards", 1, 1),
        rows,
        {"N_phi": 2, "N_pi": 2, "N_P": 0, "updates": 1},
    )
    for name, parameter in change.named_parameters():
        assert parameter.grad is None and parameter not in ctx.optimizer.state
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
