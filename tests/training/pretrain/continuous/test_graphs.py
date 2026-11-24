"""Independent nonlinear VJP oracle for overlapping continuous observations."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mcpref.models.perception.continuous.state import ContinuousState, GDNState
from mcpref.training.pretrain.continuous import graphs
from mcpref.training.pretrain.continuous.state import Stream, detached


class NonlinearPhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Parameter(torch.tensor([[0.2, -0.1], [0.3, 0.4]]))
        self.recurrent = nn.Parameter(torch.tensor([[0.4, 0.1], [-0.2, 0.3]]))
        self.readout = nn.Parameter(torch.tensor([0.7, -0.4]))
        self.observations = nn.Parameter(torch.linspace(0.01, 0.24, 24).view(12, 2))
        self.unused = nn.Parameter(torch.tensor(0.9))
        self.cfg, self.sprite, self.calls = SimpleNamespace(), None, 0
        self.double()

    def groups(self):
        return {"phi": list(self.parameters())}


def prepare(phi, frame, line, origin):
    x = phi.vision.new_tensor([line["value"] / 10, origin["value"] / 13])
    return SimpleNamespace(x=x, index=line["value"] - 1, origin=dict(origin))


def encode(phi, prepared):
    phi.calls += 1
    value = torch.sin(phi.vision @ prepared.x) + phi.observations[prepared.index]
    return value * (0.6 + torch.rand_like(value))


def decode(phi, features, prepared, state):
    layers = []
    for i in range(2):
        old = (
            state.layers[i]
            if state
            else GDNState(torch.zeros_like(features), torch.zeros_like(features))
        )
        conv = torch.tanh(features + old.conv * 0.2 + old.recurrent * 0.1)
        recurrent = torch.tanh(
            phi.recurrent @ old.recurrent + conv + features * (i + 1)
        )
        recurrent = recurrent * (0.7 + 0.2 * torch.rand_like(recurrent))
        layers.append(GDNState(conv, recurrent))
    z = (layers[0].recurrent + layers[1].recurrent * 0.3) * phi.readout
    return z, ContinuousState(tuple(layers), prepared.origin)


def forward(phi, frame, line, state):
    prepared = prepare(phi, frame, line, line if state is None else state.origin)
    return decode(phi, encode(phi, prepared), prepared, state)


@pytest.fixture
def nonlinear_graphs(monkeypatch):
    for name, value in (("prepare", prepare), ("encode", encode), ("decode", decode)):
        monkeypatch.setattr(graphs, name, value)


def assert_nested(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested(a, b)
    elif hasattr(expected, "__dataclass_fields__"):
        for key in expected.__dataclass_fields__:
            assert_nested(getattr(actual, key), getattr(expected, key))
    else:
        assert actual == expected


def oracle_event(phi, entries, anchor, item, k):
    entries.append((item, torch.get_rng_state().clone()))
    state, latents = anchor, {}
    for index, (row, rng) in enumerate(entries):
        if index == len(entries) - 1:
            z, state = forward(phi, row.frame, row.line, state)
        else:
            with torch.random.fork_rng():
                torch.set_rng_state(rng)
                z, state = forward(phi, row.frame, row.line, state)
        latents[row.id] = z
        if index == 0:
            first = detached(state)
    if len(entries) == k:
        anchor = first
        entries.pop(0)
    return latents, state, anchor


def run(k, batch_size, cached):
    phi = NonlinearPhi()
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(device="cpu", tbptt_states=k), modules={"phi": phi}
    )
    optimizer = torch.optim.SGD(phi.parameters(), lr=0.031)
    stream, anchor, entries, reports, previous = Stream(), None, [], [], None
    torch.manual_seed(713)
    for start in range(0, 12, batch_size):
        cache = graphs.SlidingGraphs(ctx) if cached else None
        values = []
        for i in range(start, min(start + batch_size, 12)):
            if i in (0, 7, 10):
                if cache:
                    cache.flush()
                stream, anchor, entries, previous = Stream(), None, [], None
            item = SimpleNamespace(id=str(i), frame=None, line={"value": i + 1})
            if cache:
                z = cache.observation(stream, item)
                latents = {key: cache.graphs[key].latent for key in cache.active}
                state, anchor = stream.recurrent, stream.anchor
            else:
                latents, state, anchor = oracle_event(phi, entries, anchor, item, k)
                z = latents[item.id]
            values.append(z.detach())
            if i not in (2, 8):
                # Independent losses, multiple readout latents and interleaved RNG.
                loss = (z.square() * torch.rand_like(z)).sum()
                loss.backward(retain_graph=True)
                old = z if previous is None else latents.get(str(i - 1), previous)
                ((z - old).square().sum() * 0.17 + z.sin().sum()).backward()
            else:
                torch.rand_like(z)
            if cache:
                cache.backward(stream)
                assert len(cache.graphs) <= k - 1
            previous, stream = z.detach(), stream.detach()
        if cache:
            cache.flush()
            assert not cache.graphs
        reports.append(
            (
                values,
                [None if p.grad is None else p.grad.clone() for p in phi.parameters()],
                detached(state),
                anchor,
                torch.get_rng_state(),
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return reports, copy.deepcopy(phi.state_dict()), phi.calls


@pytest.mark.parametrize("k,batch_size", [(1, 1), (1, 7), (4, 1), (4, 3), (4, 7)])
def test_all_parameter_gradients_rng_and_evolving_memory(
    nonlinear_graphs, k, batch_size
):
    expected, expected_weights, calls = run(k, batch_size, False)
    actual, weights, cached_calls = run(k, batch_size, True)
    assert_nested(actual, expected)
    assert_nested(weights, expected_weights)
    assert cached_calls <= calls
    if k > 1 and batch_size > 1:
        assert cached_calls < calls
