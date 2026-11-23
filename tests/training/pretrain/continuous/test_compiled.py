"""Preference and diagnostic backward preserve a real compiled FP8 phi graph."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from models.perception.world import backbone
from torch.utils.checkpoint import checkpoint
from training.pretrain.continuous.engine_world import (
    Index,
    config,
    context,
    endpoint,
    modules,
    observations,
)

from mcpref.models.perception.precision import enable_fp8_mlp
from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.state import Stream


class Phi(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        original = backbone(torch.float32).to(device)
        enable_fp8_mlp(original)
        self.mlp = original.language_model.layers[0].mlp
        if device == "cpu":
            self.mlp.compile(backend="aot_eager", fullgraph=True, dynamic=True)
        self.cfg, self.sprite = SimpleNamespace(), None

    def groups(self):
        return {"phi": list(self.parameters())}


def forward(phi, frame, line, state=None):
    x = next(phi.parameters()).new_full((1, 1, 32), line["value"] / 10)
    if state is not None:
        x = x + torch.nn.functional.pad(state, (0, 30))[None, None] * 0.5
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        z = checkpoint(phi.mlp, x, use_reentrant=False)[0, 0, :2].float()
    return z, z


class Diagnostics:
    def before_backward(self, context, components, events):
        grads = torch.autograd.grad(
            sum(components.values()),
            tuple(context.modules["phi"].parameters()),
            retain_graph=True,
        )
        assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.timeout(180)
def test_fp8_preference_and_own_backward_match_eager(
    tmp_path, monkeypatch, diagnostics, device
):
    monkeypatch.setattr(steps, "observation_step", forward)
    edge = {"id": "edge", "a": "old", "b": "new", "labels": {"inc": 0, "p": 0.8}}
    data = Index({}, [edge], {})
    rows = observations(
        "x",
        [1, 2, 3],
        endpoints={0: [endpoint("old", ["x0"], 1)], 1: [endpoint("new", ["x1"], 1)]},
    )
    results = []
    for eager in (True, False):
        model = {key: value.to(device) for key, value in modules().items()}
        model["phi"] = Phi(device)
        cfg = config(
            device=device,
            tbptt_states=2,
            batch_size=2,
            gradient_diagnostics=diagnostics,
        )
        root = tmp_path / str(eager)
        ctx = context(root, cfg, model, data)
        archive = Archive(root / "archive/shards", 1, 1)
        try:
            with torch.compiler.set_stance("force_eager") if eager else nullcontext():
                stream, result = steps.window(
                    ctx,
                    Stream(),
                    archive,
                    rows[:2],
                    {"N_phi": 2, "N_pi": 2, "N_P": 1, "updates": 2},
                    Diagnostics() if diagnostics else None,
                )
                assert result.counts == {"phi": 1, "ce": 2, "shift": 1, "P": 1}
                results.append(
                    {
                        name: {key: p.clone() for key, p in m.state_dict().items()}
                        for name, m in model.items()
                    }
                )
                ctx.step = 2
                # Next ordinary observation may compile again under updated weights.
                _, result = steps.window(
                    ctx,
                    stream,
                    archive,
                    rows[2:],
                    {"N_phi": 2, "N_pi": 2, "N_P": 1, "updates": 2},
                    Diagnostics() if diagnostics else None,
                )
                assert result.counts == {"phi": 1, "ce": 0, "shift": 1, "P": 0}
                assert all(
                    torch.isfinite(p).all()
                    for m in model.values()
                    for p in m.parameters()
                )
        finally:
            archive.close()
    for name in results[0]:
        for key, value in results[0][name].items():
            torch.testing.assert_close(value, results[1][name][key], rtol=0, atol=0)
