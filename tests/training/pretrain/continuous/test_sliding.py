"""Per-event gradient support survives independent optimizer boundaries."""

import pytest
import torch
from training.pretrain.continuous.engine_world import config, context, observations

from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.archive import Archive
from mcpref.training.pretrain.continuous.state import Stream


@pytest.mark.parametrize("batch_size", [1, 3, 4, 7])
def test_every_loss_has_sliding_support_across_updates(
    tmp_path, monkeypatch, batch_size
):
    cfg = config(tbptt_states=4, batch_size=batch_size)
    ctx = context(tmp_path, cfg)
    phi = ctx.modules["phi"]
    phi.weight = torch.nn.Parameter(torch.linspace(0.1, 0.9, 9))
    ctx = context(tmp_path, cfg, ctx.modules)
    rows = observations("x", range(1, 10))
    supports = []

    def forward(phi, frame, line, state):
        z = phi.weight[line["value"] - 1] + (0 if state is None else state * 0.5)
        return z, z

    def losses(context, stream, observation, z, result):
        return {"phi": z.square() / 2}

    class Capture:
        def before_backward(self, context, components, events):
            (gradient,) = torch.autograd.grad(
                components["phi"], phi.weight, retain_graph=True
            )
            supports.append(gradient.nonzero().flatten().tolist())

    monkeypatch.setattr(steps, "observation_step", forward)
    monkeypatch.setattr(steps, "own_losses", losses)
    population = {"N_phi": 9, "N_pi": 0, "N_P": 0, "updates": 9}
    archive, stream = Archive(tmp_path / "archive/shards", 1, 1), Stream()
    try:
        for start in range(0, len(rows), batch_size):
            ctx.step = start // batch_size + 1
            stream, _ = steps.window(
                ctx,
                stream,
                archive,
                rows[start : start + batch_size],
                population,
                Capture(),
            )
    finally:
        archive.close()
    assert supports == [list(range(max(0, t - 3), t + 1)) for t in range(9)]


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_delayed_shift_support(tmp_path, monkeypatch, k, batch_size):
    cfg = config(tbptt_states=k, batch_size=batch_size, w_phi=0.0, w_change=0.0)
    ctx = context(tmp_path, cfg)
    ctx.modules["phi"].weight = torch.nn.Parameter(torch.linspace(0.1, 0.9, 9))
    ctx = context(tmp_path, cfg, ctx.modules, total=100)
    rows = observations("x", range(1, 10))
    supported = []

    def forward(phi, frame, line, state):
        z = phi.weight[line["value"] - 1] * torch.tensor([1.0, 0.5])
        z = z + (0 if state is None else state * 0.5)
        return z, z

    original_ce = steps.policy_ce

    def no_ce_gradient(*args):
        loss, change, pending, row = original_ce(*args)
        return loss.detach(), change, pending, row

    class Capture:
        def before_backward(self, context, components, events):
            loss = components.get("pi")
            if loss is not None:
                (gradient,) = torch.autograd.grad(
                    loss,
                    context.modules["phi"].weight,
                    retain_graph=True,
                    allow_unused=True,
                )
                supported.append(
                    [] if gradient is None else gradient.nonzero().flatten().tolist()
                )

    monkeypatch.setattr(steps, "observation_step", forward)
    monkeypatch.setattr(steps, "policy_ce", no_ce_gradient)
    archive, stream = Archive(tmp_path / "archive/shards", 1, 1), Stream()
    try:
        for start in range(0, len(rows), batch_size):
            ctx.step = start // batch_size + 1
            stream, _ = steps.window(
                ctx,
                stream,
                archive,
                rows[start : start + batch_size],
                {"N_phi": 8, "N_pi": 8, "N_P": 0, "updates": 1},
                Capture(),
            )
    finally:
        archive.close()
    assert supported == [list(range(max(0, t - k + 1), t)) for t in range(1, 9)]
