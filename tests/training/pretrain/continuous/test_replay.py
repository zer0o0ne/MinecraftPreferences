"""Closed-form recurrent oracles for refreshed memory, accumulation and RNG."""

from types import SimpleNamespace

import pytest
import torch

from mcpref.training.pretrain.continuous.replay import observation
from mcpref.training.pretrain.continuous.state import Stream


def scalar_context(k):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(0.4)
    return SimpleNamespace(
        cfg=SimpleNamespace(device="cpu", tbptt_states=k), modules={"phi": model}
    )


def item(t):
    return SimpleNamespace(id=str(t), frame=None, line={"x": (t + 1) / 10})


def linear(model, frame, line, state):
    old = model.weight.new_zeros(()) if state is None else state
    z = model.weight.squeeze() * old + line["x"]
    return z, z


@pytest.mark.parametrize("k,batch_size", [(1, 3), (4, 1), (4, 3), (4, 7)])
def test_values_anchor_and_accumulated_gradient_match_closed_form(k, batch_size):
    ctx, stream = scalar_context(k), Stream()
    weight = ctx.modules["phi"].weight
    optimizer = torch.optim.SGD([weight], lr=0.01)
    anchor, expected_gradient = 0.0, 0.0
    for t in range(11):
        a = weight.detach().item()
        xs = [(j + 1) / 10 for j in range(max(0, t - k + 1), t + 1)]
        m = len(xs)
        # h = a^m h0 + sum_{i=0}^{m-1} a^(m-1-i) x_i, with h0 stop-grad.
        value = a**m * anchor + sum(a ** (m - 1 - i) * x for i, x in enumerate(xs))
        derivative = m * a ** (m - 1) * anchor + sum(
            (m - 1 - i) * a ** (m - 2 - i) * x for i, x in enumerate(xs[:-1])
        )
        z = observation(ctx, stream, item(t), linear)
        torch.testing.assert_close(z, z.new_tensor(value))
        (z.square() / 2).backward()
        expected_gradient += value * derivative
        torch.testing.assert_close(
            weight.grad, weight.new_tensor([[expected_gradient]])
        )
        if m == k:
            anchor = a * anchor + xs[0]
            torch.testing.assert_close(stream.anchor, z.new_tensor(anchor))
        stream = stream.detach()
        if (t + 1) % batch_size == 0 or t == 10:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            expected_gradient = 0.0


def test_replay_preserves_original_draws_and_global_rng():
    def stochastic(model, frame, line, state):
        z, _ = linear(model, frame, line, state)
        z = z * torch.rand((), dtype=z.dtype)
        return z, z

    runs = []
    for k in (1, 4):
        ctx, stream, values = scalar_context(k), Stream(), []
        torch.manual_seed(72)
        for t in range(9):
            z = observation(ctx, stream, item(t), stochastic)
            values.append(z.detach())
            z.backward()
            stream = stream.detach()
        runs.append((torch.stack(values), torch.get_rng_state()))
    for a, b in zip(*runs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
