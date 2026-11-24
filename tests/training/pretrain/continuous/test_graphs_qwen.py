"""Production local VJPs match sliding replay with real Qwen and retained FLA graphs."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from models.perception.continuous.test_functional import position, text_model
from models.perception.continuous.test_gpu import assert_fla
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.training.pretrain.continuous import graphs, replay
from mcpref.training.pretrain.continuous.state import Stream, detached


class Perception(torch.nn.Module):
    def __init__(self, device, dtype, checkpointed):
        super().__init__()
        self.text = text_model(dtype, device).train()
        self.text.layers[1].self_attn.attention_dropout = 0.2
        if checkpointed:
            self.text.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.frontend = torch.nn.Linear(32, 32).to(device=device, dtype=dtype)
            self.head = torch.nn.Linear(32, 4).to(device=device, dtype=dtype)
        self.device, self.mixed = device, device == "cuda" and dtype == torch.float32
        generator = torch.Generator(device=device).manual_seed(42)
        self.sources = [
            torch.randn(1, length, 32, device=device, dtype=dtype, generator=generator)
            for length in (5, 63, 65)
        ]
        self.encoded = {}


def prepare(phi, frame, line, origin):
    return SimpleNamespace(index=line["index"], origin=origin)


def encode(phi, prepared):
    with torch.autocast(phi.device, dtype=torch.bfloat16, enabled=phi.mixed):
        out = phi.frontend(phi.sources[prepared.index % 3])
        out = torch.nn.functional.dropout(out, p=0.2, training=phi.training)
    out.retain_grad()
    phi.encoded[prepared.index] = out
    return out


def decode(phi, embeddings, prepared, state):
    with torch.autocast(phi.device, dtype=torch.bfloat16, enabled=phi.mixed):
        out, states = text_step(
            phi.text,
            embeddings,
            position(embeddings),
            () if state is None else state.layers,
        )
    return phi.head(out[0, -1].to(phi.head.weight.dtype)), ContinuousState(
        states, prepared.origin
    )


def forward(phi, frame, line, state):
    prepared = prepare(phi, frame, line, line if state is None else state.origin)
    return decode(phi, encode(phi, prepared), prepared, state)


def snapshot(phi, stream, loss, rng):
    return detached((loss, stream.recurrent, stream.anchor, rng)), [
        None if p.grad is None else p.grad.detach().clone() for p in phi.parameters()
    ]


def assert_tree(actual, expected, tolerance):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, **tolerance)
    elif hasattr(expected, "__dataclass_fields__"):
        for key in expected.__dataclass_fields__:
            assert_tree(getattr(actual, key), getattr(expected, key), tolerance)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree(a, b, tolerance)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key], tolerance)
    else:
        assert actual == expected


def run(phi, batch, initial_rng, local, learning_rate=0.0001):
    device, stream = torch.device(phi.device), Stream()
    context = SimpleNamespace(
        cfg=SimpleNamespace(device=phi.device, tbptt_states=4), modules={"phi": phi}
    )
    optimizer = torch.optim.SGD(phi.parameters(), lr=learning_rate)
    engine, events, updates, feature_gradients, old = None, [], [], [], {}
    accumulated = {}
    with replay.replay_rng(initial_rng, device):
        for t in range(2 * batch + 1):
            if t % batch == 0:
                accumulated = {}
                if local:
                    engine = graphs.SlidingGraphs(context)
                    old = {}
            phi.encoded.clear()
            row = SimpleNamespace(id=str(t), frame=t, line={"index": t})
            z = (
                engine.observation(stream, row)
                if local
                else replay.observation(context, stream, row, forward)
            )
            if local:
                old.update(engine.graphs)
                before = {key: detached(g.embeddings.grad) for key, g in old.items()}
            stream.remember(row.id, z, 6)
            loss = sum(
                (i + 1) * value.float().square().mean()
                for i, value in enumerate(stream.recent.values())
            )
            loss.backward()
            if local:
                engine.backward(stream)
                for key, graph in old.items():
                    if int(key) < max(0, t - 3):
                        assert_tree(
                            graph.embeddings.grad, before[key], {"rtol": 0, "atol": 0}
                        )
            else:
                for key, value in phi.encoded.items():
                    gradient = value.grad.detach().float()
                    accumulated[key] = accumulated.get(key, 0) + gradient
            rng = replay.capture_rng(device)
            events.append(snapshot(phi, stream, loss, rng)[0])
            stream = stream.detach()
            if (t + 1) % batch == 0 or t == 2 * batch:
                if local:
                    engine.flush()
                    accumulated = {
                        int(key): graph.embeddings.grad.detach().float()
                        for key, graph in old.items()
                    }
                feature_gradients.append(detached(accumulated))
                updates.append(snapshot(phi, stream, loss, rng)[1])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return events, updates, feature_gradients, detached(phi.state_dict())


def assert_gradient_direction(actual, expected, limit):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.isfinite(a).all()
            assert bool(torch.count_nonzero(a)) == bool(torch.count_nonzero(b))
    a = torch.cat([x.detach().double().flatten() for x in actual if x is not None])
    b = torch.cat([x.detach().double().flatten() for x in expected if x is not None])
    relative = (a - b).norm() / b.norm()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    assert relative < limit, float(relative)
    assert cosine > 1 - limit**2 / 2, float(cosine)


def assert_local_qwen(
    monkeypatch,
    device,
    dtype,
    checkpointed,
    batch,
    full_precision=False,
    learning_rate=0.0001,
):
    for name in ("prepare", "encode", "decode"):
        monkeypatch.setattr(graphs, name, globals()[name])
    phi = Perception(device, dtype, checkpointed)
    if full_precision:
        phi.mixed = False
    reference = copy.deepcopy(phi)
    with torch.random.fork_rng(devices=[device] if device == "cuda" else []):
        torch.manual_seed(177)
        rng = replay.capture_rng(torch.device(device))
    expected = run(reference, batch, rng, local=False, learning_rate=learning_rate)
    actual = run(phi, batch, rng, local=True, learning_rate=learning_rate)
    tolerance = (
        {"rtol": 3e-4, "atol": 3e-5}
        if device == "cpu" or full_precision
        else {"rtol": 0.025, "atol": 0.025}
    )
    exact_events = len(actual[0]) if learning_rate == 0 else batch
    assert_tree(
        actual[0][:exact_events], expected[0][:exact_events], {"rtol": 0, "atol": 0}
    )
    for a, b in zip(actual[0], expected[0]):
        assert_tree(a[3], b[3], {"rtol": 0, "atol": 0})
    if device == "cpu":
        assert_tree(actual, expected, tolerance)
    else:
        assert_tree(actual[0], expected[0], tolerance)
        weights = torch.cat([x.double().flatten() for x in actual[3].values()])
        target = torch.cat([x.double().flatten() for x in expected[3].values()])
        assert (weights - target).norm() / target.norm() < (
            1e-6 if full_precision else 0.001
        )
        # Gradient direction is compared at identical weights. Real updates are
        # checked separately because BF16 rounding changes the later trajectories.
        count = len(actual[1]) if learning_rate == 0 or full_precision else 1
        limit = 1e-5 if full_precision else 0.01
        for a, b in zip(actual[1][:count], expected[1][:count]):
            assert_gradient_direction(a, b, limit)
        for values in actual[1]:
            assert all(x is None or torch.isfinite(x).all() for x in values)
        for a, b in zip(actual[2][:count], expected[2][:count]):
            assert a.keys() == b.keys()
            assert_gradient_direction(list(a.values()), [b[key] for key in a], limit)
    assert any(g is not None and g.abs().sum() > 0 for g in actual[1][0])


@runs_backbone
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("batch", [2, 7])
def test_cpu_real_qwen_local_vjps(monkeypatch, checkpointed, batch):
    assert_local_qwen(monkeypatch, "cpu", torch.float32, checkpointed, batch)


@pytest.mark.gpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("batch", [2, 7])
def test_cuda_real_qwen_retained_fla_vjps(monkeypatch, dtype, checkpointed, batch):
    assert_fla()
    assert_local_qwen(monkeypatch, "cuda", dtype, checkpointed, batch, learning_rate=0)
    torch.cuda.synchronize()


@pytest.mark.gpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_quantized_update_trajectories(monkeypatch, dtype):
    assert_fla()
    assert_local_qwen(monkeypatch, "cuda", dtype, True, 7)
    torch.cuda.synchronize()


@pytest.mark.gpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("checkpointed", [False, True])
def test_cuda_full_precision_proves_deferred_vjp_graph(monkeypatch, checkpointed):
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    monkeypatch.setattr(
        module,
        "torch_chunk_gated_delta_rule",
        module.torch_chunk_gated_delta_rule.__wrapped__,
    )
    assert_local_qwen(
        monkeypatch, "cuda", torch.float32, checkpointed, 7, full_precision=True
    )
    torch.cuda.synchronize()
