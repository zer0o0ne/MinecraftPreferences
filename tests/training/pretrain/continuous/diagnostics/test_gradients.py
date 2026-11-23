"""Real truncated gradients and chronology select the reported diagnostics."""

import pytest
import torch
from torch.nn import functional as F
from training.pretrain.continuous.data.test_training import (
    model,
)
from training.pretrain.continuous.data.test_training import (
    observation_step as artifact_step,
)
from training.pretrain.continuous.data.world import make_world
from training.pretrain.continuous.diagnostics.gradient_oracle import (
    Capture,
    assert_close,
)
from training.pretrain.continuous.engine_world import (
    Index,
    Reader,
    config,
    context,
    endpoint,
    modules,
    observation_step,
    observations,
)
from training.pretrain.continuous.test_resume import world

from mcpref.training.pretrain.continuous import loop, steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.data import index as index_builder
from mcpref.training.pretrain.continuous.data.reader import Index as DiskIndex
from mcpref.training.pretrain.continuous.diagnostics import gradients, selection
from mcpref.training.pretrain.continuous.schedule import epoch_updates
from mcpref.training.pretrain.continuous.state import Stream


@pytest.mark.parametrize("K", [1, 2])
def test_components_and_update_dots_match_autograd_without_changing_training(
    tmp_path, monkeypatch, K
):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    monkeypatch.setattr(
        gradients,
        "gradient_steps",
        lambda c: {step: ["test"] for step in range(1, c.total + 1)},
    )
    cfg, data = config(tbptt_states=K, epochs=1, gradient_diagnostics=True), world(K)
    total = epoch_updates(data, cfg)
    plain = context(tmp_path / "plain", cfg, modules(dropout=0.3), data, total)
    plain.root.mkdir()
    torch.manual_seed(39)
    loop.train(plain)
    expected_rng = torch.get_rng_state().clone()
    captured = context(tmp_path / "captured", cfg, modules(dropout=0.3), data, total)
    captured.root.mkdir()
    torch.manual_seed(39)
    hook = Capture()
    loop.train(captured, hook)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, module in plain.modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(
                value, captured.modules[name].state_dict()[key], rtol=0, atol=0
            )
    assert len(hook.records) == total
    assert any(record["counts"]["P"] == 0 for record in hook.records.values())
    for record in hook.records.values():
        if not record["counts"]["P"]:
            assert (
                record["phi"]["total"]["norms"]["P"]
                == record["P"]["total"]["norm"]
                == 0
            )
            assert record["nonfinite"]["phi"]["P"] is None


@pytest.mark.parametrize("K,derivative", [(1, 3.0), (2, 3.5)])
def test_reported_preference_gradient_detaches_old_endpoint_and_tbptt_history(
    tmp_path, monkeypatch, K, derivative
):
    monkeypatch.setattr(steps, "observation_step", observation_step)
    target_step = 2 if K == 1 else 1
    monkeypatch.setattr(gradients, "gradient_steps", lambda _: {target_step: ["test"]})
    cfg = config(tbptt_states=K, gradient_diagnostics=True)
    edge = {"id": "edge", "a": "old", "b": "new", "labels": {"inc": 0, "p": 0.8}}
    pop = {"N_phi": 0, "N_pi": 0, "N_P": 1, "updates": target_step}
    data = Index({}, [edge], pop)
    ctx = context(tmp_path, cfg, index=data, total=target_step)
    rows = observations(
        "x",
        [1, 3],
        actions=[[]],
        endpoints={0: [endpoint("old", ["x0"], 1)], 1: [endpoint("new", ["x1"], 1)]},
    )
    w = ctx.modules["phi"].weight.detach().clone().requires_grad_()
    old = w.detach()[None, None]
    current = (derivative * w + (3.5 - derivative) * w.detach())[None, None]
    s, t = ctx.modules["P"](old, current)
    loss = (
        target_step
        * cfg.w_P
        * (
            2 * F.binary_cross_entropy_with_logits(t, torch.zeros_like(t))
            + 2 * F.binary_cross_entropy_with_logits(s, torch.full_like(s, 0.8))
        )
    )
    expected = torch.autograd.grad(loss, w)[0].norm().item()
    hook = Capture()
    hook.start(ctx)
    archive, stream = Archive(tmp_path / "archive/shards", 1, 1), Stream()
    try:
        for start in range(0, 2, K):
            ctx.step = start // K + 1
            stream, _ = steps.window(
                ctx, stream, archive, rows[start : start + K], pop, hook
            )
    finally:
        archive.close()
    assert_close(hook.records[target_step]["phi"]["total"]["norms"]["P"], expected)


@pytest.mark.parametrize("K", [1, 3])
def test_selection_matches_actual_preference_event_steps(tmp_path, monkeypatch, K):
    fixture = make_world(tmp_path, monkeypatch, K=K)
    index_builder.run(fixture.cfg, fixture.path)
    monkeypatch.setattr(selection, "REPO", tmp_path)
    monkeypatch.setattr(selection, "load_config", lambda _: None)
    monkeypatch.setattr(selection, "load_masks", lambda _: fixture.masks)
    monkeypatch.setattr(steps, "observation_step", artifact_step)
    monkeypatch.setenv("MCPREF_PROFILE", "")
    data = DiskIndex(fixture.out)
    cfg = config(
        tbptt_states=K, epochs=3, gradient_diagnostics=True, action_mask="mask.yaml"
    )
    total = epoch_updates(data, cfg) * cfg.epochs
    ctx = context(tmp_path / "run", cfg, model(fixture), data, total)
    ctx.root.mkdir()
    seen = []

    class Events:
        def before_backward(self, context, components, events):
            if events.get("preference"):
                seen.append(context.step)

    try:
        selected = selection.gradient_steps(ctx)
        loop.train(ctx, Events())
    finally:
        data.close()
    candidates = sorted(set(seen))
    middle = next(
        (step for step in candidates if step >= max(1, total // 2)), candidates[-1]
    )
    expected = {}
    for role, step in (
        ("beginning", candidates[0]),
        ("middle", middle),
        ("end", candidates[-1]),
    ):
        expected.setdefault(step, []).append(role)
    assert selected == expected
