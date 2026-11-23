"""Real Qwen sliding graphs survive updates and checkpoint recomputation."""

from types import SimpleNamespace

import torch
from models.perception.continuous.test_functional import inputs, position, text_model
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous.functional import text_step
from mcpref.models.perception.continuous.state import ContinuousState
from mcpref.training.pretrain.continuous.replay import observation
from mcpref.training.pretrain.continuous.state import Stream


def assert_sliding(device, dtype):
    model = text_model(dtype, device).train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(device=device, tbptt_states=4), modules={"phi": model}
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    chunks, stream, old_inputs = inputs(device, dtype), Stream(), []
    active = []

    def forward(phi, frame, line, state):
        x = frame.detach().clone().requires_grad_()
        active.append((line["index"], x))
        out, memory = text_step(
            phi, x, position(x), () if state is None else state.layers
        )
        return out[0, -1], ContinuousState(memory, {})

    for t in range(6):
        active.clear()
        row = SimpleNamespace(id=str(t), frame=chunks[t % 3], line={"index": t})
        z = observation(ctx, stream, row, forward)
        loss = (z.float() * torch.arange(z.numel(), device=device)).sum()
        current_inputs = [x for _, x in active]
        gradients = torch.autograd.grad(
            loss, old_inputs + current_inputs, allow_unused=True, retain_graph=True
        )
        assert [i for i, _ in active] == list(range(max(0, t - 3), t + 1))
        assert all(g is None for g in gradients[: len(old_inputs)])
        assert all(
            g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
            for g in gradients[len(old_inputs) :]
        )
        loss.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        stream = stream.detach()
        old_inputs = current_inputs
        if (t + 1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


@runs_backbone
def test_checkpointed_cpu_sliding_replay_crosses_optimizer_updates():
    assert_sliding("cpu", torch.float32)
