"""Optimized steps retain real own/P event ordering, pending targets and AdamW."""

import copy
from math import ceil
from types import SimpleNamespace

import pytest
import torch
from training.pretrain.continuous.engine_world import (
    Index,
    config,
    context,
    endpoint,
    modules,
    observations,
)
from training.pretrain.continuous.test_graphs import (
    NonlinearPhi,
    assert_nested,
    forward,
    nonlinear_graphs,
)

from mcpref.training.pretrain.continuous import graphs, steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.state import Stream, detached

__all__ = ["nonlinear_graphs"]


def data():
    edges = [
        {"id": a + b, "a": a, "b": b, "labels": labels}
        for a, b, labels in (
            ("A", "B", {"inc": 0, "p": 0.8}),
            ("A", "C", {"inc": 1, "p": None}),
            ("B", "D", {"inc": 0, "p": 0.1}),
            ("C", "D", {"inc": 0, "p": 0.4}),
        )
    ]
    rows = observations(
        "x",
        list(range(1, 8)),
        actions=[["a"], [], ["?"], ["a", "b"], ["b"], ["a"]],
        endpoints={
            1: [endpoint("A", ["x0", "x1"], 2)],
            4: [endpoint("B", ["x3", "x4"], 2)],
        },
    ) + observations(
        "y",
        list(range(8, 13)),
        actions=[["b"], ["a"], [], ["a"]],
        endpoints={
            1: [endpoint("C", ["y0", "y1"], 2)],
            4: [endpoint("D", ["y3", "y4"], 2)],
        },
    )
    return rows, Index({}, edges, {})


class Capture:
    def __init__(self):
        self.gradients = []

    def before_update(self, ctx, result):
        self.gradients.append(
            {
                name: [
                    None if p.grad is None else p.grad.clone()
                    for p in module.parameters()
                ]
                for name, module in ctx.modules.items()
            }
        )


def run(root, cfg, model, cached, monkeypatch):
    rows, index = data()
    population = {
        "N_phi": 7,
        "N_pi": 7,
        "N_P": 4,
        "updates": ceil(len(rows) / cfg.batch_size),
    }
    ctx = context(root, cfg, copy.deepcopy(model), index, total=population["updates"])
    capture, stream, reports = Capture(), Stream(), []
    archive = Archive(root / "archive/shards", 1, 1)
    monkeypatch.setattr(
        steps, "reusable", graphs.SlidingGraphs if cached else lambda _: None
    )
    torch.manual_seed(83)
    try:
        for start in range(0, len(rows), cfg.batch_size):
            ctx.step = start // cfg.batch_size + 1
            stream, result = steps.window(
                ctx,
                stream,
                archive,
                rows[start : start + cfg.batch_size],
                population,
                capture,
            )
            reports.append(
                (
                    detached(stream),
                    result,
                    copy.deepcopy(ctx.optimizer.state_dict()),
                    {
                        name: copy.deepcopy(module.state_dict())
                        for name, module in ctx.modules.items()
                    },
                    torch.get_rng_state().clone(),
                    sorted(archive.seen),
                    sorted(archive.consumed),
                )
            )
        assert sorted(archive.consumed) == ["AB", "AC", "BD", "CD"]
        assert not archive.live and stream.pending is None
    finally:
        archive.close()
    return reports, capture.gradients


@pytest.mark.parametrize("k,batch_size", [(1, 1), (1, 3), (4, 1), (4, 3), (4, 7)])
@pytest.mark.parametrize("frozen", [False, True])
def test_real_losses_updates_and_archives_match_replay(
    tmp_path, monkeypatch, nonlinear_graphs, k, batch_size, frozen
):
    model = modules(dropout=0.3)
    model["phi"] = NonlinearPhi()
    for module in model.values():
        module.double()
    if frozen:
        model["phi"].requires_grad_(False)
    cfg = config(tbptt_states=k, batch_size=batch_size, preference_micro_batch=1)
    monkeypatch.setattr(steps, "observation_step", forward)
    expected = run(tmp_path / "baseline", cfg, model, False, monkeypatch)
    actual = run(tmp_path / "reused", cfg, model, True, monkeypatch)
    assert_nested(actual, expected)


@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("event", [0, 3, 4, 7])
def test_last_loss_reaches_exactly_k_observation_parameters(nonlinear_graphs, k, event):
    phi = NonlinearPhi()
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(device="cpu", tbptt_states=k), modules={"phi": phi}
    )
    cache, stream = graphs.SlidingGraphs(ctx), Stream()
    for t in range(event + 1):
        row = SimpleNamespace(id=str(t), frame=None, line={"value": t + 1})
        z = cache.observation(stream, row)
        if t == event:
            z.sum().backward()
        cache.backward(stream)
        stream = stream.detach()
    cache.flush()
    support = phi.observations.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert support == list(range(max(0, event - k + 1), event + 1))
