"""Serialized shared shards and graph drive the real loop, archive and checkpoint path."""

import pytest
import torch
from training.pretrain.continuous.data.world import make_world
from training.pretrain.continuous.engine_world import config, context, modules
from training.pretrain.continuous.test_resume import Stop, compare

from mcpref.training.pretrain.continuous import loop, steps
from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index


def model(world):
    result = modules(dropout=0.3)
    result["phi"].cfg = world.phi_cfg
    pi = result["pi"]
    pi.names = world.masks.names
    pi.index = {name: i for i, name in enumerate(pi.names)}
    pi.project = torch.nn.Linear(2 + len(pi.names) + 1, 2)
    pi.output = torch.nn.Linear(2, len(pi.names))
    return result


def observation_step(phi, frame, line, state=None):
    z = phi.weight * (line["xpos"] - 99) + (0 if state is None else 0.5 * state)
    return z, z


@pytest.mark.parametrize("K,step,when", [(1, 6, "staged"), (3, 4, "published")])
def test_shared_artifacts_train_and_resume_with_exact_populations(
    tmp_path, monkeypatch, K, step, when
):
    world = make_world(tmp_path, monkeypatch, K=K)
    index.run(world.cfg, world.path)
    data = Index(world.out)
    cfg = config(tbptt_states=K, checkpoint_steps=2)
    total = data.run["populations"]["train"]["M"] * cfg.epochs
    monkeypatch.setattr(steps, "observation_step", observation_step)
    monkeypatch.setenv("MCPREF_PROFILE", "")
    try:
        full = context(tmp_path / "full", cfg, model(world), data, total)
        full.root.mkdir()
        torch.manual_seed(29)
        loop.train(full)
        partial = context(tmp_path / "resumed", cfg, model(world), data, total)
        partial.root.mkdir()
        torch.manual_seed(29)
        with pytest.raises(InterruptedError):
            loop.train(partial, Stop(step, when))
        resumed = context(partial.root, cfg, model(world), data, total)
        torch.manual_seed(827)
        loop.train(resumed)
        compare(full.root, resumed.root)
    finally:
        data.close()
