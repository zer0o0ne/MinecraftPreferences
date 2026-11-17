"""Reuse the saved-record runtime world for real MCTS gameplay tests."""

import pytest
import torch
import yaml
from runtime.sampling.conftest import site as site  # noqa: PLC0414
from training.rl.learning.helpers import BASE

PLAY = {
    "mode": "pretrained",
    "trajectories": 2,
    "out_dir": "data/play-test",
    "device": "cpu",
    "sampling": "configs/runtime/sampling.yaml",
    "rl_cycle": "configs/training/rl/cycle.yaml",
}


@pytest.fixture
def gameplay(site, monkeypatch):
    path = site.root / "configs/runtime/play.yaml"
    path.write_text(yaml.safe_dump(PLAY))
    rl = site.root / PLAY["rl_cycle"]
    rl.write_text(yaml.safe_dump(BASE | {"sampling": "wrong.yaml", "device": "cuda"}))
    model = site.root / "data/pretrain-scaled-init/model.pt"
    built = {name: getattr(site.modules, name) for name in ("phi", "pi", "P")}
    torch.save({name: module.state_dict() for name, module in built.items()}, model)
    monkeypatch.setattr(
        "mcpref.runtime.sampling.models.build_modules", lambda *a, **kw: built
    )
    monkeypatch.setattr(
        "mcpref.runtime.sampling.models.load_masks", lambda cfg: object()
    )
    site.env.close = lambda: site.events.append(("close", site.env.index))
    monkeypatch.setattr("mcpref.runtime.play.run.Env", lambda cfg: site.env)
    site.play_path = path
    return site
