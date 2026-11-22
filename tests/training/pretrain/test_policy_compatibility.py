"""Old temporal objectives must not silently resume with newly aligned targets."""

import json
from types import SimpleNamespace

import pytest
import torch

from mcpref.infrastructure.runs.files import write_json
from mcpref.training.pretrain import checkpoints
from mcpref.training.pretrain.classical import checks
from mcpref.training.pretrain.continuous import checkpoints as continuous_checkpoints
from mcpref.training.pretrain.continuous import loop
from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index
from training.pretrain.continuous.data.world import make_world
from training.pretrain.continuous.test_startup import files


@pytest.mark.parametrize("alignment", [None, "outgoing-action-v0"])
def test_classical_old_run_rejected_before_reading_inputs(
    tmp_path, monkeypatch, alignment
):
    monkeypatch.setattr(checks, "REPO", tmp_path)
    path = tmp_path / "train/run.json"
    write_json(path, {"complete": True, "policy_alignment": alignment})
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="Policy alignment changed"):
        checks.header(SimpleNamespace(out_dir="train"), tmp_path / "missing.yaml")
    assert path.read_bytes() == original


def test_continuous_completed_old_run_is_not_reused(tmp_path, monkeypatch):
    cfg, path, data = files(tmp_path, monkeypatch)
    run_path = tmp_path / cfg.out_dir / "run.json"
    write_json(run_path, {"complete": True})
    original = run_path.read_bytes()
    with pytest.raises(RuntimeError, match="Policy alignment changed"):
        loop.pretrain(cfg, path, {}, data)
    assert run_path.read_bytes() == original


@pytest.mark.parametrize(
    "artifact", ["model.pt", "joint-model.pt", "checkpoints", "retrain"]
)
def test_training_artifact_without_header_cannot_acquire_new_alignment(
    tmp_path, artifact
):
    path = tmp_path / artifact
    path.write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="Policy alignment changed"):
        checkpoints.check_alignment(tmp_path / "run.json")
    assert path.read_bytes() == b"legacy"
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("alignment", [None, "outgoing-action-v0"])
def test_old_checkpoint_rejected_before_loading_weights(
    tmp_path, continuous, alignment
):
    directory = tmp_path / "checkpoints/000000"
    directory.mkdir(parents=True)
    # Deliberately omit modules/optimizer: the alignment check must precede loads.
    payload = {"policy_alignment": alignment}
    if continuous:
        payload["pretrain_mode"] = "continuous"
    torch.save(payload, directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="Checkpoint policy alignment differs"):
        if continuous:
            continuous_checkpoints.restore(SimpleNamespace(root=tmp_path))
        else:
            checkpoints.restore(tmp_path / "checkpoints", 1, {})


def test_old_continuous_index_population_requires_rebuild(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    index.run(world.cfg, world.path)
    path = world.out / "run.json"
    header = json.loads(path.read_text())
    assert header["format"] == "mcpref-continuous-index-v2"
    write_json(path, header | {"format": "mcpref-continuous-index-v1"})
    before = {p: p.read_bytes() for p in world.out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="complete continuous training index"):
        Index(world.out)
    with pytest.raises(ValueError, match="inputs changed"):
        index.run(world.cfg, world.path)
    assert {p: p.read_bytes() for p in before} == before
