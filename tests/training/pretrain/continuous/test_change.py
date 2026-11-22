"""Independent occurrence-level gradients for continuous policy change learning."""

from types import SimpleNamespace

import pytest
import torch
from training.pretrain.continuous.engine_world import config, context, observations

from mcpref.training.pretrain.continuous.events import WindowResult
from mcpref.training.pretrain.continuous.losses.own import policy_change
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.pretrain.continuous.steps import own_losses


@pytest.mark.parametrize("previous", [["a"], ["a", "a", "a"]])
def test_each_current_occurrence_uses_previous_membership_not_frequency(previous):
    logits = torch.tensor([[0.3, -0.8]], dtype=torch.float64, requires_grad=True)
    probability = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    pi = SimpleNamespace(names=("a", "b"), change=lambda _: probability)
    incoming = {"known": True, "actions": previous}
    q = logits.new_tensor([2 / 3, 1 / 3])
    loss = policy_change(pi, None, logits, q, incoming, True)
    losses = -logits.detach().log_softmax(-1)[0]
    expected = (2 * probability.detach()[0] * losses[0] + 0.8 * losses[1]) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    occurrence_weights = logits.new_tensor([2 * 0.2 / 3, 0.8 / 3])
    expected_logits = logits.detach().softmax(-1) * occurrence_weights.sum()
    expected_logits -= occurrence_weights
    torch.testing.assert_close(logits.grad, expected_logits)
    torch.testing.assert_close(
        probability.grad, ((2 * losses[0] - losses[1]) / 3)[None]
    )


@pytest.mark.parametrize("kind", ["first", "unknown", "gap", "disabled"])
def test_unknown_predecessor_or_zero_weight_skips_only_auxiliary(
    tmp_path, monkeypatch, kind
):
    ctx = context(tmp_path, config(w_change=0.0 if kind == "disabled" else 1.0))
    first, current, _ = observations("x", [1, 2, 3], actions=[["a"], ["b"]])
    stream = Stream()
    if kind != "first":
        incoming = first.interval | {"valid": False, "known": kind != "unknown"}
        if kind == "gap":
            incoming["b"] = "different-state"
        stream.previous = {"interval": incoming}

    def unexpected(_):
        raise AssertionError("excluded auxiliary must not build its graph")

    monkeypatch.setattr(ctx.modules["pi"], "change", unexpected)
    result = WindowResult()
    own = own_losses(ctx, stream, current, torch.tensor([0.4, -0.2]), result)
    assert "pi_change" not in own and "pi" in own
    assert result.counts["ce"] == 1 and result.sums["pi_change"] == 0
    assert stream.pending["valid"]


def test_known_idle_predecessor_is_all_change_on_the_same_dropout_readout(
    tmp_path, monkeypatch
):
    ctx, stream = context(tmp_path), Stream()
    first, current, _ = observations("x", [1, 2, 3], actions=[[], ["b", "b"]])
    stream.previous = {"interval": first.interval}
    pi, calls = ctx.modules["pi"], []
    original = pi.readout

    def readout(Y, C):
        value = original(Y, C)
        calls.append(value)
        return value

    monkeypatch.setattr(pi, "readout", readout)
    result = WindowResult()
    own = own_losses(ctx, stream, current, torch.tensor([0.4, -0.2]), result)
    assert len(calls) == 1
    probability = pi.change(calls[0][:, -1])
    torch.testing.assert_close(own["pi_change"], (1 - probability).sum() * own["pi"])
    assert result.sums["pi_change"] > 0


def test_actual_previous_membership_survives_input_dropout_and_previous_mask(
    tmp_path, monkeypatch
):
    ctx, stream = context(tmp_path), Stream()
    first, current, _ = observations("x", [1, 2, 3], actions=[["b"], ["b"]])
    stream.previous = {
        "interval": first.interval | {"valid": False, "legal_actions": []}
    }
    pi, original = ctx.modules["pi"], ctx.modules["pi"].readout
    captured = []

    def dropped(Y, C):
        empty = torch.zeros_like(C)
        empty[..., -1] = 1
        value = original(Y, empty)
        captured.append(value)
        return value

    monkeypatch.setattr(pi, "readout", dropped)
    own = own_losses(ctx, stream, current, torch.tensor([0.4, -0.2]), WindowResult())
    probability = pi.change(captured[0][:, -1])
    torch.testing.assert_close(own["pi_change"], probability.sum() * own["pi"])
