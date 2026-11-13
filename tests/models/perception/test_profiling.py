"""Real tiny Qwen produces named vision/language ranges and actual batch sizes."""

import json
from types import SimpleNamespace

import pytest
import torch

from mcpref.infrastructure.profiling.runtime import TrainingProfile
from mcpref.infrastructure.profiling.scopes import phase
from mcpref.models.perception.phi import Phi
from mcpref.models.perception.precision import enable_fp8_mlp
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)

pytestmark = runs_backbone


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_qwen_profile_has_real_ranges_and_preserves_gradients(
    tmp_path, monkeypatch, batch, compiled
):
    path = tmp_path / "profile.yaml"
    path.write_text("skip_steps: 0\nsteps: 1\ntop_ops: 5\n")
    monkeypatch.setenv("MCPREF_PROFILE", str(path))
    phi = Phi(backbone(torch.bfloat16), tokenizer(), config(tmp_path)).bfloat16()
    if compiled:
        enable_fp8_mlp(phi.backbone)
        for layer in phi.backbone.language_model.layers:
            layer.mlp.compile(backend="aot_eager", fullgraph=True, dynamic=True)
    clips = [
        random_clip(i, list(range(i + 1)), [False] * (i + 1)) for i in range(batch)
    ]
    states = phi.prepare(clips)
    reference = phi.batch_frames_z(states)
    sum(z.float().square().sum() for z in reference).backward()
    grads = {
        name: p.grad.clone() for name, p in phi.named_parameters() if p.grad is not None
    }
    phi.zero_grad(set_to_none=True)
    cfg = SimpleNamespace(
        device="cpu", batch=batch, micro_batch=batch, gradient_diagnostics=False
    )
    with (
        TrainingProfile(cfg, tmp_path, {"phi": phi}, "qwen") as profile,
        profile.step(1, batch),
    ):
        actual = phi.batch_frames_z(states)
        with phase("backward"):
            sum(z.float().square().sum() for z in actual).backward()
    for z, other in zip(actual, reference, strict=True):
        torch.testing.assert_close(z, other, rtol=0, atol=0)
    for name, p in phi.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    report = json.loads(
        next(tmp_path.glob("profiles/qwen/*/*/summary.json")).read_text()
    )
    names = {region["name"] for region in report["regions"]}
    assert {
        "qwen.preprocess",
        "qwen.transfer",
        "qwen.forward",
        "qwen.vision",
        "qwen.language",
        "phi.head",
        "backward",
    } <= names
    assert len(report["qwen_batches"]) == 1
    row = report["qwen_batches"][0]
    assert row["rows"] == batch and row["frames"] == sum(range(1, batch + 1))
    assert (row["padding_fraction"] > 0) == (batch > 1)
    for module in (phi.backbone.visual, phi.backbone.language_model):
        assert not module._forward_hooks and not module._forward_pre_hooks
