"""Independent transition oracle shared by pretraining, sampling and RL."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mcpref.datasets.clips.io import Clip
from mcpref.gui.mask import Masks, clip_masks
from mcpref.models.policy.config import PolicyConfig
from mcpref.models.policy.loss import PolicyLoss, cross_entropy
from mcpref.models.policy.network import Policy
from mcpref.models.policy.targets import next_actions, targets
from mcpref.runtime.sampling.history import append
from mcpref.runtime.sampling.shards import RawHistory
from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.events import WindowResult
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.rl.learning.history import latent_history

NAMES = ("a", "b", "c")
Z = torch.tensor(
    [[0.0, 0.0], [1.0, 2.0], [4.0, -1.0], [13.0, 5.0]], dtype=torch.float64
)
# The three distinct actions cause these three distinct transitions, respectively.
DELTAS = torch.tensor([[1.0, 2.0], [3.0, -3.0], [9.0, 6.0]], dtype=torch.float64)
INPUTS = torch.eye(4, dtype=torch.float64)[[3, 0, 1, 2]]


def world(history=2, actions=NAMES, *, device="cpu", dtype=torch.float64):
    masks = Masks(
        NAMES,
        np.ones(3, dtype=bool),
        {str(i): np.eye(3, dtype=bool)[i % 3] for i in range(4)},
        90.0,
    )
    clip = Clip(
        np.zeros((4, 6), dtype=np.uint8),
        2,
        2,
        {
            "id": "causal",
            "states": list(range(4)),
            "actions": [[a] for a in actions],
            "n": [1, 1, 1],
            "gui": [True] * 4,
            "classes": list(map(str, range(4))),
            "pitch": [0.0] * 4,
        },
        [{"hotbar": 0}] * 4,
    )
    cfg = PolicyConfig(
        history=history,
        width=8,
        heads=2,
        hidden=12,
        layers=1,
        shift_hidden=10,
        rope_base=1000.0,
        norm_eps=1e-6,
        action_dropout=0.0,
        shift_weight=0.7,
        seed=43,
        latency_ms=1.0,
        latency_fail_ms=2.0,
        latency_calls=1,
    )
    pi = Policy(cfg, 2, NAMES).to(device=device, dtype=dtype).eval()
    loss = PolicyLoss(cfg, masks, lambda x: torch.linalg.vector_norm(x, dim=-1))
    return pi, loss, clip


def continuous(pi, loss, clip, latents, monkeypatch):
    # Perception's independent objective is outside this policy alignment test.
    monkeypatch.setattr(
        steps,
        "phi_interval",
        lambda module, previous, z, observation: ({"total": z.sum() * 0}, {}),
    )
    allowed = clip_masks(loss.masks, clip.info, clip.lines)
    intervals = [
        {
            "a": str(t),
            "b": str(t + 1),
            "actions": names,
            "n": len(names),
            "known": True,
            "valid": any(allowed[t, pi.index[a]] for a in names),
        }
        for t, names in enumerate(clip.info["actions"])
    ]
    intervals.append(
        {"a": "3", "b": None, "actions": [], "n": 0, "known": False, "valid": False}
    )
    ctx = SimpleNamespace(modules={"pi": pi, "pi_loss": loss, "phi_loss": None})
    stream, result, histories, readouts = Stream(), WindowResult(), [], []
    for t, z in enumerate(latents):
        observation = SimpleNamespace(
            id=str(t),
            interval=intervals[t],
            mask=allowed[t],
            source="test",
            frame=None,
            meta=None,
        )
        steps.own_losses(ctx, stream, observation, z, result)
        histories.append(torch.stack([row["c"] for row in stream.policy]))
        readouts.append(None if stream.pending is None else stream.pending["readout"])
        stream.previous = {
            "interval": intervals[t],
            "frame": None,
            "meta": None,
            "z": z,
        }
    assert stream.pending is None
    return result, histories, readouts


@pytest.mark.parametrize(
    "history,device,dtype",
    [
        (1, "cpu", torch.float64),
        (2, "cpu", torch.float64),
        (6, "cpu", torch.float64),
        pytest.param(
            2, "cuda", torch.bfloat16, marks=[pytest.mark.gpu, pytest.mark.timeout(180)]
        ),
    ],
)
def test_same_transition_has_same_inputs_readout_action_and_shift_everywhere(
    history, device, dtype, monkeypatch
):
    pi, loss, clip = world(history, device=device, dtype=dtype)
    latents = Z.to(device=device, dtype=dtype).clone().requires_grad_()
    inputs, deltas = INPUTS.to(latents), DELTAS.to(latents)
    target_rows = torch.eye(3, dtype=dtype, device=device)
    out = pi(latents[None], [clip])
    terms, valid, diagnostic = loss.terms(out, latents[None], [clip])
    streamed, histories, readouts = continuous(pi, loss, clip, latents, monkeypatch)
    torch.testing.assert_close(out[2][0], inputs)
    torch.testing.assert_close(diagnostic["q"][0], target_rows)
    assert valid["shift_valid"].tolist() == [[True, True, True]]
    assert next_actions(clip, pi.index).tolist() == [0, 1, 2, 0]
    assert streamed.counts["ce"] == streamed.counts["shift"] == 3
    ce, shifts, sampled = [], [], None
    for t, z in enumerate(latents):
        sampled = append(sampled, z, None if t == 0 else t - 1, 3, history)
        raw = RawHistory([None] * (t + 1), [None, *range(t)], 0)
        trained = latent_history(raw, latents[: t + 1], history, 3)
        expected = inputs[max(0, t - history + 1) : t + 1]
        torch.testing.assert_close(sampled.C[0], expected)
        torch.testing.assert_close(trained.C, sampled.C)
        torch.testing.assert_close(trained.Y, sampled.Y)
        torch.testing.assert_close(histories[t], expected)
        if t == 3:
            continue
        readout = pi.readout(trained.Y, trained.C)[:, -1]
        torch.testing.assert_close(readouts[t], readout)
        logits = pi.logits(readout)
        prediction = pi.shift(readout, torch.tensor([t], device=device))
        row = streamed.ce_rows[t]
        torch.testing.assert_close(out[0][:, t], logits)
        torch.testing.assert_close(out[1][:, t], prediction)
        torch.testing.assert_close(row["logits"], logits)
        torch.testing.assert_close(row["targets"][0], target_rows[t])
        assert row["action"] == t and row["shift_valid"]
        assert row["masks"].tolist() == [i == t for i in range(3)]
        ce.append(-torch.log_softmax(logits, -1)[0, t])
        shifts.append(0.7 * (prediction[0] - deltas[t]).square().sum())
        torch.testing.assert_close(streamed.shift_rows[t]["total"], shifts[-1])
    expected_ce, expected_shift = torch.stack(ce).mean(), torch.stack(shifts).mean()
    torch.testing.assert_close(terms["action"][0], expected_ce)
    torch.testing.assert_close(terms["shift"][0], expected_shift)
    torch.testing.assert_close(streamed.sums["pi_action"] / 3, expected_ce)
    torch.testing.assert_close(streamed.sums["pi_shift"] / 3, expected_shift)
    variables = [latents, *pi.parameters()]
    oracle_gradients = torch.autograd.grad(expected_ce + expected_shift, variables)
    terms["total"].sum().backward()
    for variable, oracle in zip(variables, oracle_gradients, strict=True):
        assert variable.grad is not None and torch.isfinite(variable.grad).all()
        if dtype == torch.float64:
            torch.testing.assert_close(variable.grad, oracle)
    assert torch.count_nonzero(latents.grad)
    assert torch.count_nonzero(pi.head.weight.grad)
    assert torch.count_nonzero(pi.dynamics.w2.weight.grad)
    assert torch.count_nonzero(latents.grad[-1]) == 0  # factual successor is detached


@pytest.mark.parametrize("t", [0, 1])
def test_future_action_cannot_change_current_policy_input_or_ce(t, monkeypatch):
    actions = list(NAMES)
    actions[t + 1] = NAMES[t]
    pi, loss, clip = world()
    _, _, changed = world(actions=actions)
    outputs, streams, ce = [], [], []
    for candidate in (clip, changed):
        out = pi(Z[None], [candidate])
        q, _ = targets(candidate, loss.masks)
        outputs.append(out)
        streams.append(continuous(pi, loss, candidate, Z, monkeypatch)[0])
        ce.append(cross_entropy(out[0][:, :-1], torch.tensor(q, dtype=Z.dtype)[None]))
    torch.testing.assert_close(outputs[0][2][:, : t + 1], outputs[1][2][:, : t + 1])
    torch.testing.assert_close(outputs[0][0][:, t], outputs[1][0][:, t])
    torch.testing.assert_close(outputs[0][1][:, t], outputs[1][1][:, t])
    torch.testing.assert_close(ce[0][:, t], ce[1][:, t])
    for key in ("logits", "targets", "previous"):
        torch.testing.assert_close(
            streams[0].ce_rows[t][key], streams[1].ce_rows[t][key]
        )
