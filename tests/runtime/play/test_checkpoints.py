"""Gameplay takes fixed weights from one published checkpoint of ongoing training."""

import hashlib
import json

import pytest
import torch
import yaml
from runtime.play.conftest import PLAY

from mcpref.config import read_yaml
from mcpref.runtime.play import run
from mcpref.runtime.sampling.shards import load_step


def select(gameplay, directory):
    path = gameplay.root / PLAY["sampling"]
    path.write_text(yaml.safe_dump(read_yaml(path) | {"checkpoint_dir": directory}))


def test_live_checkpoint_is_loaded_and_survives_source_pruning(gameplay, monkeypatch):
    source_dir = gameplay.root / "data/ongoing/checkpoints/000031"
    source_dir.mkdir(parents=True)
    source = source_dir / "checkpoint.pt"
    expected = {}
    for name in ("phi", "pi", "P"):
        module = getattr(gameplay.modules, name)
        expected[name] = {
            key: torch.full_like(value, 0.25)
            for key, value in module.state_dict().items()
        }
    torch.save({"modules": expected, "optimizer": {}, "step": 3200}, source)
    original = source.read_bytes()
    pretrain = gameplay.root / "data/pretrain-scaled-init"
    (pretrain / "run.json").write_text('{"complete": false}')
    (pretrain / "model.pt").unlink()
    select(gameplay, "data/ongoing/checkpoints/000031")
    configs_before = {
        p: p.read_bytes() for p in (gameplay.root / "configs").rglob("*.yaml")
    }

    def env(cfg):
        source.unlink()  # Training is allowed to prune after the launch snapshot.
        return gameplay.env

    monkeypatch.setattr(run, "Env", env)
    out = run.run(gameplay.play_path)
    assert json.loads((out / "run.json").read_text())["completed_trajectories"] == 2
    assert load_step(out / "sampling", 3).after.line["id"] == 4
    assert (out / "checkpoint/checkpoint.pt").read_bytes() == original
    assert json.loads((out / "checkpoint/source.json").read_text()) == {
        "path": str(source),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    effective = read_yaml(out / "configs/sampling.yaml")
    assert effective["checkpoint_dir"] == f"{PLAY['out_dir']}/checkpoint"
    sampled = json.loads((out / "sampling/run.json").read_text())
    assert sampled["sha256"]["pretrain_model"] == hashlib.sha256(original).hexdigest()
    for name, weights in expected.items():
        actual = getattr(gameplay.modules, name)
        assert not actual.training
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    assert json.loads((pretrain / "run.json").read_text()) == {"complete": False}
    assert all(p.read_bytes() == content for p, content in configs_before.items())


@pytest.mark.parametrize("kind", ["missing", "staged", "both", "pruned"])
def test_unusable_checkpoint_rejected_before_environment(gameplay, monkeypatch, kind):
    source_dir = gameplay.root / (
        "data/000003.part" if kind == "staged" else "data/000003"
    )
    source_dir.mkdir()
    if kind in ("staged", "both"):
        (source_dir / "checkpoint.pt").write_bytes(b"not loaded")
    if kind == "both":
        (source_dir / "model.pt").write_bytes(b"not loaded")
    if kind == "pruned":
        (source_dir / "losses.json").write_text("[]")
    select(gameplay, source_dir.relative_to(gameplay.root).as_posix())
    monkeypatch.setattr(run, "Env", lambda cfg: pytest.fail("environment constructed"))
    with pytest.raises((ValueError, FileNotFoundError)):
        run.run(gameplay.play_path)
    metadata = json.loads((gameplay.root / PLAY["out_dir"] / "run.json").read_text())
    assert metadata["status"] == "failed" and metadata["completed_trajectories"] == 0
