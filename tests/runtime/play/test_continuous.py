"""Continuous fixed-weight gameplay survives config materialization and snapshots."""

import json

import pytest
import torch
import yaml
from runtime.play.conftest import PLAY
from runtime.sampling.continuous_world import setup

from mcpref.config import read_yaml
from mcpref.models.perception.continuous import ContinuousState
from mcpref.models.perception.continuous.state import GDNState
from mcpref.runtime.play import run
from mcpref.training.pretrain.continuous.state import Stream
from mcpref.training.pretrain.modes import selected_config


@pytest.mark.parametrize("checkpoint", [False, True])
def test_continuous_gameplay_preserves_mode_and_memory(
    gameplay, monkeypatch, checkpoint
):
    setup(gameplay, monkeypatch)
    if checkpoint:
        directory = gameplay.root / "data/ongoing/checkpoints/000003"
        directory.mkdir(parents=True)
        source = directory / "checkpoint.pt"
        torch.save(
            {
                "pretrain_mode": "continuous",
                "modules": {
                    name: getattr(gameplay.modules, name).state_dict()
                    for name in ("phi", "pi", "P")
                },
                "stream": Stream(
                    recurrent=ContinuousState(
                        (GDNState(torch.zeros(1), torch.tensor(900.0)),), {"id": -100}
                    )
                ),
            },
            source,
        )
        metadata = directory / "metadata.json"
        metadata.write_text(json.dumps({"pretrain_mode": "continuous"}))
        path = gameplay.root / PLAY["sampling"]
        path.write_text(
            yaml.safe_dump(
                read_yaml(path)
                | {"checkpoint_dir": directory.relative_to(gameplay.root).as_posix()}
            )
        )

        def env(cfg):
            source.unlink()
            metadata.unlink()
            return gameplay.env

        monkeypatch.setattr(run, "Env", env)
    output = run.run(gameplay.play_path)
    sampling = read_yaml(output / "configs/sampling.yaml")
    mode, _ = selected_config(gameplay.root / sampling["pretrain"])
    assert mode == "continuous"
    saved = torch.load(output / "sampling/shards/000001/state.pt", weights_only=True)
    assert saved["stream"]["origin"] == {"id": 0}
    assert saved["stream"]["layers"][0]["recurrent"].item() == 15
    assert [event[1] for event in gameplay.events if event[0] == "stream"] == list(
        range(5)
    )
    assert gameplay.env.resets == 1
    if checkpoint:
        assert json.loads((output / "checkpoint/metadata.json").read_text()) == {
            "pretrain_mode": "continuous"
        }


def test_continuous_rl_materializes_replay_settings(gameplay, monkeypatch):
    from mcpref.runtime.play.config import load_config
    from mcpref.runtime.play.configs import create_directory, materialize
    from mcpref.runtime.sampling.config import load_config as load_sampling
    from mcpref.training.rl.config import ContinuousConfig
    from mcpref.training.rl.config import load_config as load_rl
    from mcpref.training.rl.context import load_context

    setup(gameplay, monkeypatch)
    gameplay.play_path.write_text(yaml.safe_dump(PLAY | {"mode": "rl"}))
    rl_path = gameplay.root / PLAY["rl_cycle"]
    rl_path.write_text(
        yaml.safe_dump(
            read_yaml(rl_path)
            | {"continuous": {"gradient_history": 4, "replay_stride": 64}}
        )
    )
    cfg = load_config(gameplay.play_path)
    effective = materialize(cfg, create_directory(cfg))
    rl = load_rl(gameplay.root / effective.rl_cycle)
    sampling = load_sampling(gameplay.root / rl.sampling)
    context = load_context(rl, sampling)
    assert context.mode == "continuous"
    assert context.states == 5
    assert context.replay == ContinuousConfig(4, 64)


def test_continuous_rl_requires_replay_config_before_world_construction(
    gameplay, monkeypatch
):
    setup(gameplay, monkeypatch)
    gameplay.play_path.write_text(yaml.safe_dump(PLAY | {"mode": "rl"}))
    monkeypatch.setattr(run, "Env", lambda cfg: pytest.fail("world constructed"))
    with pytest.raises(ValueError, match="continuous replay config"):
        run.run(gameplay.play_path)
