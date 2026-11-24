"""Ephemeral reused graphs reconstruct exactly from ordinary saved stream state."""

import pytest
import torch
from training.pretrain.continuous.engine_world import Reader, config, context, modules
from training.pretrain.continuous.test_graphs import NonlinearPhi, nonlinear_graphs
from training.pretrain.continuous.test_replay_resume import exact_equal
from training.pretrain.continuous.test_resume import Stop, world

from mcpref.training.pretrain.continuous import graphs, loop, steps

__all__ = ["nonlinear_graphs"]


def model():
    result = modules(dropout=0.3)
    result["phi"] = NonlinearPhi()
    for module in result.values():
        module.double()
    return result


@pytest.mark.parametrize("batch_size,stop", [(1, 3), (3, 1), (3, 2), (7, 1)])
def test_resume_rebuilds_graphs_without_changing_training(
    tmp_path, monkeypatch, nonlinear_graphs, batch_size, stop
):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "reusable", graphs.SlidingGraphs)
    cfg = config(tbptt_states=4, batch_size=batch_size, keep_checkpoints=20)
    index = world(4)
    full = context(tmp_path / "full", cfg, model(), index)
    full.root.mkdir()
    torch.manual_seed(83)
    loop.train(full)
    interrupted = context(tmp_path / "resumed", cfg, model(), index)
    interrupted.root.mkdir()
    torch.manual_seed(83)
    with pytest.raises(InterruptedError):
        loop.train(interrupted, Stop(stop, "published"))
    resumed = context(interrupted.root, cfg, model(), index)
    torch.manual_seed(999)
    loop.train(resumed)
    for path in full.root.glob("checkpoints/*/checkpoint.pt"):
        expected = torch.load(path, weights_only=False)
        actual = torch.load(
            resumed.root / path.relative_to(full.root), weights_only=False
        )
        for key in (
            "modules",
            "optimizer",
            "position",
            "stream",
            "torch_rng",
            "cuda_rng",
        ):
            assert exact_equal(actual[key], expected[key]), (key, path)
    assert exact_equal(
        torch.load(full.root / "model.pt", weights_only=True),
        torch.load(resumed.root / "model.pt", weights_only=True),
    )
