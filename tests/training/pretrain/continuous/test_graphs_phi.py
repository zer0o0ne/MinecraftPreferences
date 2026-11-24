"""The production Phi selects bounded graphs and honors checkpoint configuration."""

import copy
from types import SimpleNamespace

import pytest
import torch
from models.perception.continuous.test_gpu import assert_fla
from models.perception.continuous.test_observation import world
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous import observation_step
from mcpref.models.perception.precision import enable_fp8_mlp
from mcpref.training.pretrain.continuous import graphs, replay
from mcpref.training.pretrain.continuous.state import Stream, detached


def execute(phi, rows, checkpointed, diagnostics, learning_rate=0.00001):
    cfg = SimpleNamespace(
        device=str(next(phi.parameters()).device),
        tbptt_states=4,
        activation_checkpointing=checkpointed,
        gradient_diagnostics=diagnostics,
    )
    context = SimpleNamespace(cfg=cfg, modules={"phi": phi})
    optimizer = torch.optim.SGD(phi.parameters(), lr=learning_rate)
    stream, records, events = Stream(), [], []
    for start in range(0, len(rows), 2):
        cache = graphs.reusable(context)
        assert (cache is None) == diagnostics
        assert phi.backbone.is_gradient_checkpointing == checkpointed
        for row in rows[start : start + 2]:
            z = (
                replay.observation(context, stream, row, observation_step)
                if cache is None
                else cache.observation(stream, row)
            )
            (z.square().mean() + z.sin().mean()).backward()
            if cache is not None:
                cache.backward(stream)
            events.append(
                detached(
                    (
                        z,
                        stream.recurrent,
                        stream.anchor,
                        replay.capture_rng(torch.device(cfg.device)),
                    )
                )
            )
            stream = stream.detach()
        if cache is not None:
            cache.flush()
        records.append(
            [None if p.grad is None else p.grad.clone() for p in phi.parameters()]
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return records, phi.state_dict(), stream.recurrent, stream.anchor, events


@runs_backbone
@pytest.mark.parametrize("checkpointed", [False, True])
def test_real_multimodal_gradients_and_updates_match_diagnostic_replay(
    tmp_path, checkpointed
):
    from training.pretrain.continuous.test_graphs_qwen import assert_tree

    phi, frames, lines = world(tmp_path)
    rows = [
        SimpleNamespace(id=str(i), frame=f, line=line)
        for i, (f, line) in enumerate(zip(frames, lines))
    ]
    expected = execute(copy.deepcopy(phi), rows, checkpointed, True)
    actual = execute(phi, rows, checkpointed, False)
    # Qwen's normalization and recurrent accumulation use FP32 even in double.
    assert_tree(actual, expected, {"rtol": 1e-5, "atol": 1e-8})


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=runs_backbone),
        pytest.param("cuda", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.timeout(180)
def test_fp8_local_graphs_survive_retained_vjps_and_weight_updates(tmp_path, device):
    phi, frames, lines = world(tmp_path)
    phi = phi.float().to(device)
    enable_fp8_mlp(phi.backbone)
    rows = [
        SimpleNamespace(id=str(i), frame=f, line=line)
        for i, (f, line) in enumerate(zip(frames, lines))
    ]
    before = {key: value.clone() for key, value in phi.state_dict().items()}
    with torch.compiler.set_stance("force_eager"):
        gradients, weights, *_ = execute(phi, rows, False, False)
    assert all(
        g is None or torch.isfinite(g).all() for batch in gradients for g in batch
    )
    assert all(torch.isfinite(value).all() for value in weights.values())
    assert any(not torch.equal(value, before[key]) for key, value in weights.items())


@pytest.mark.gpu
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("learning_rate", [0.0, 0.00001])
def test_cuda_real_phi_grouped_embeddings_match_replay(
    tmp_path, dtype, checkpointed, learning_rate
):
    from training.pretrain.continuous.test_graphs_qwen import (
        assert_gradient_direction,
        assert_tree,
    )

    assert_fla()
    phi, frames, lines = world(tmp_path)
    phi = phi.to(device="cuda", dtype=dtype).train()
    phi.backbone.language_model.layers[1].self_attn.attention_dropout = 0.2
    rows = [
        SimpleNamespace(id=str(i), frame=frames[i % 3], line=lines[i % 3])
        for i in range(7)
    ]
    reference = copy.deepcopy(phi)
    with torch.random.fork_rng(devices=["cuda"]):
        torch.manual_seed(179)
        rng = replay.capture_rng(torch.device("cuda"))
        with replay.replay_rng(rng, torch.device("cuda")):
            expected = execute(reference, rows, checkpointed, True, learning_rate)
        with replay.replay_rng(rng, torch.device("cuda")):
            actual = execute(phi, rows, checkpointed, False, learning_rate)
    exact = len(rows) if learning_rate == 0 else 2
    assert_tree(actual[4][:exact], expected[4][:exact], {"rtol": 0, "atol": 0})
    for a, b in zip(actual[4], expected[4]):
        assert_tree(a[3], b[3], {"rtol": 0, "atol": 0})
    assert_tree(actual[2:], expected[2:], {"rtol": 0.025, "atol": 0.025})
    count = len(actual[0]) if learning_rate == 0 else 1
    for a, b in zip(actual[0][:count], expected[0][:count]):
        assert_gradient_direction(a, b, 0.01)
    weight = phi.backbone.get_input_embeddings().weight
    index = next(i for i, p in enumerate(phi.parameters()) if p is weight)
    for a, b in zip(actual[0][:count], expected[0][:count]):
        assert_gradient_direction([a[index]], [b[index]], 0.01)
        assert torch.equal(a[index].ne(0).any(dim=1), b[index].ne(0).any(dim=1))
    assert all(
        g is None or torch.isfinite(g).all() for batch in actual[0] for g in batch
    )
    assert_gradient_direction(
        list(actual[1].values()), list(expected[1].values()), 0.001
    )
    torch.cuda.synchronize()
