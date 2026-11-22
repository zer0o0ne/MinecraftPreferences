"""Causal policy inputs and targets use the action of the same transition."""

from dataclasses import replace

import numpy as np
import torch
from training.pretrain.continuous.engine_world import context, observations

from mcpref.training.pretrain.continuous.events import WindowResult
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.pretrain.continuous.steps import own_losses


def test_policy_input_contains_only_the_action_that_arrived_at_this_state(tmp_path):
    ctx, stream, result = context(tmp_path), Stream(), WindowResult()
    rows = observations("x", [1, 2, 3], actions=[["b"], ["a"]])
    z = torch.tensor([0.4, -0.2], requires_grad=True)
    own_losses(ctx, stream, rows[0], z, result)
    torch.testing.assert_close(stream.policy[0]["c"], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(result.ce_rows[0]["targets"], torch.tensor([[0.0, 1.0]]))
    stream.previous = {
        "interval": rows[0].interval,
        "frame": rows[0].frame,
        "meta": rows[0].meta,
        "z": z,
    }
    own_losses(ctx, stream, rows[1], 2 * z, result)
    torch.testing.assert_close(stream.policy[-1]["c"], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(result.ce_rows[1]["targets"], torch.tensor([[1.0, 0.0]]))


def test_one_interval_trains_ce_and_its_own_shift_without_another_action(tmp_path):
    ctx, stream, result = context(tmp_path), Stream(), WindowResult()
    first, last = observations("x", [1, 2], actions=[["b"]])
    z, successor = torch.tensor([0.4, -0.2]), torch.tensor([0.7, -0.8])
    own = own_losses(ctx, stream, first, z, result)
    assert "pi" in own and result.counts["ce"] == 1
    assert stream.pending["action"] == 1 and stream.pending["valid"]
    pi = ctx.modules["pi"]
    readout = pi.readout(z[None, None], torch.tensor([[[0.0, 0.0, 1.0]]]))[:, -1]
    prediction = pi.shift(readout, torch.tensor([1]))
    expected = 0.3 * (prediction - (successor - z)[None]).square().sum()
    own = own_losses(ctx, stream, last, successor, result)
    torch.testing.assert_close(own["pi"], expected)
    assert result.counts["ce"] == result.counts["shift"] == 1
    assert stream.pending is None


def test_policy_target_uses_current_state_mask_and_not_next_interval(tmp_path):
    ctx, stream, result = context(tmp_path), Stream(), WindowResult()
    first = observations("x", [1, 2, 3], actions=[["a", "b"], ["b"]])[0]
    first = replace(first, mask=np.array([True, False]))
    own_losses(ctx, stream, first, torch.tensor([0.4, -0.2]), result)
    torch.testing.assert_close(result.ce_rows[0]["targets"], torch.tensor([[1.0, 0.0]]))
    assert not stream.pending["valid"]
    assert result.ce_rows[0]["masks"].tolist() == [True, False]


def test_multiaction_incoming_interval_does_not_disable_current_shift(tmp_path):
    ctx, stream, result = context(tmp_path), Stream(), WindowResult()
    first, current, _ = observations("x", [1, 2, 3], actions=[["a", "b"], ["b"]])
    stream.previous = {
        "interval": first.interval,
        "frame": first.frame,
        "meta": first.meta,
        "z": torch.tensor([0.4, -0.2]),
    }
    own_losses(ctx, stream, current, torch.tensor([0.7, -0.8]), result)
    torch.testing.assert_close(stream.policy[-1]["c"], torch.tensor([0.5, 0.5, 0.0]))
    assert stream.pending["valid"] and stream.pending["action"] == 1
