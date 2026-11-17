"""Real pretrained MCTS and failures in one caller-owned environment."""

import json

import pytest
import torch
import yaml
from runtime.play.conftest import PLAY

from mcpref.runtime.play import run
from mcpref.runtime.sampling.shards import load_step


def test_default_launch_plays_two_trajectories_with_fixed_weights(
    gameplay, monkeypatch, capsys
):
    monkeypatch.setattr(
        run, "run_cycle", lambda *a: pytest.fail("RL in pretrained mode")
    )
    monkeypatch.chdir(gameplay.root / "configs")
    modules = [getattr(gameplay.modules, name) for name in ("phi", "pi", "P")]
    weights = [[p.detach().clone() for p in module.parameters()] for module in modules]
    source_model = gameplay.root / "data/pretrain-scaled-init/model.pt"
    before = source_model.read_bytes()
    assert run.main([]) == 0
    out = gameplay.root / PLAY["out_dir"]
    metadata = json.loads((out / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["completed_trajectories"] == 2
    assert gameplay.env.resets == 1 and gameplay.env.index == 4
    assert gameplay.events[-1] == ("close", 4)
    assert len([e for e in gameplay.events if e[0] == "readout"]) >= 4
    assert load_step(out / "sampling", 2).before.line["id"] == 2
    assert load_step(out / "sampling", 3).after.line["id"] == 4
    assert (out / "diagnostics").is_dir()
    assert not any((out / name).exists() for name in ("rl", "pairs", "history"))
    assert not (gameplay.root / "data/diagnostics").exists()
    assert source_model.read_bytes() == before
    for module, original in zip(modules, weights, strict=True):
        for parameter, expected in zip(module.parameters(), original, strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            assert parameter.grad is None
    captured = capsys.readouterr()
    assert str(out) in captured.out and "2/2" in captured.err


@pytest.mark.parametrize("mode", ["pretrained", "rl"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_keeps_completed_count_and_closes_world(
    gameplay, monkeypatch, mode, error
):
    gameplay.play_path.write_text(yaml.safe_dump(PLAY | {"mode": mode}))
    calls = []

    def sample_once(*args):
        calls.append(args)
        if len(calls) == 2:
            raise error("stopped")

    class Sampler:
        def __init__(self, cfg, env):
            assert env is gameplay.env

        def reset(self):
            pass

        sample = staticmethod(sample_once)

    monkeypatch.setattr(run, "Sampler", Sampler)
    monkeypatch.setattr(run, "run_cycle", sample_once)
    with pytest.raises(error, match="stopped"):
        run.run(gameplay.play_path)
    metadata = json.loads((gameplay.root / PLAY["out_dir"] / "run.json").read_text())
    assert metadata["status"] == "failed" and metadata["completed_trajectories"] == 1
    assert gameplay.events[-1][0] == "close"
    if mode == "rl":
        assert all(args[2] is gameplay.env for args in calls)


def test_sampler_construction_failure_closes_world(gameplay, monkeypatch):
    def fail(*args):
        raise RuntimeError("model failed")

    monkeypatch.setattr(run, "Sampler", fail)
    assert run.main([str(gameplay.play_path)]) == 1
    assert gameplay.events[-1][0] == "close"
