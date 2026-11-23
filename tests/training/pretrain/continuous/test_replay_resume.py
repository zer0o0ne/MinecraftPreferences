"""Exact replay/RNG continuation through fragment and optimizer boundaries."""

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from training.pretrain.continuous.engine_world import Reader, config, context, modules
from training.pretrain.continuous.test_resume import Stop, world

from mcpref.training.pretrain.continuous import loop, steps


def exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return (
            type(a) == type(b)
            and len(a) == len(b)
            and all(exact_equal(x, y) for x, y in zip(a, b))
        )
    if hasattr(a, "__dataclass_fields__"):
        return type(a) == type(b) and all(
            exact_equal(getattr(a, k), getattr(b, k)) for k in a.__dataclass_fields__
        )
    return a == b


def stochastic_checkpointed_step(phi, frame, line, state):
    h = torch.zeros_like(phi.weight) if state is None else state

    def step(w, h):
        return (w * line["value"] + h * 0.5) * (torch.rand_like(w) + 0.8)

    z = checkpoint(step, phi.weight, h, use_reentrant=False)
    return z, z


@pytest.mark.parametrize(
    "batch_size,stop",
    [(2, i) for i in (1, 2, 3, 4, 5)]
    + [(3, i) for i in (1, 2, 3, 4)]
    + [(7, i) for i in (1, 2)],
)
def test_checkpointed_stochastic_resume(tmp_path, monkeypatch, batch_size, stop):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", stochastic_checkpointed_step)
    cfg = config(tbptt_states=4, batch_size=batch_size, keep_checkpoints=20)
    data = world(4)
    full = context(tmp_path / "full", cfg, modules(dropout=0.3), data)
    full.root.mkdir()
    torch.manual_seed(87)
    loop.train(full)
    interrupted = context(tmp_path / "resumed", cfg, modules(dropout=0.3), data)
    interrupted.root.mkdir()
    torch.manual_seed(87)
    with pytest.raises(InterruptedError):
        loop.train(interrupted, Stop(stop, "published"))
    resumed = context(interrupted.root, cfg, modules(dropout=0.3), data)
    torch.manual_seed(123)
    loop.train(resumed)
    for path in full.root.glob("checkpoints/*/checkpoint.pt"):
        a = torch.load(path, weights_only=False)
        b = torch.load(resumed.root / path.relative_to(full.root), weights_only=False)
        for key in (
            "modules",
            "optimizer",
            "position",
            "stream",
            "torch_rng",
            "cuda_rng",
        ):
            assert exact_equal(a[key], b[key]), (batch_size, stop, key, path)
    assert exact_equal(
        torch.load(full.root / "model.pt", weights_only=True),
        torch.load(resumed.root / "model.pt", weights_only=True),
    )
