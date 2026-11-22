"""Relocating configs preserves old index identity without accepting new settings."""

import dataclasses
import json

import pytest
from training.pretrain.continuous.data.world import make_world

from mcpref.infrastructure.runs.files import write_json
from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index


@pytest.mark.parametrize("changed", [False, True])
def test_existing_index_after_config_relocation(tmp_path, monkeypatch, changed):
    world = make_world(tmp_path, monkeypatch)
    directory = tmp_path / "configs/training/pretrain"
    directory.mkdir(parents=True)
    old_train = directory / "continuous.yaml"
    old_index = directory / "continuous-index.yaml"
    (tmp_path / "train.yaml").rename(old_train)
    cfg = dataclasses.replace(
        world.cfg, training_config=old_train.relative_to(tmp_path).as_posix()
    )
    write_json(old_index, dataclasses.asdict(cfg))
    expected = index.run(cfg, old_index)
    original = (world.out / "run.json").read_bytes()
    destination = directory / "continuous"
    destination.mkdir()
    new_train = destination / "config.yaml"
    new_index = destination / "index.yaml"
    old_train.rename(new_train)
    old_index.rename(new_index)
    relocated = dataclasses.replace(
        cfg, training_config=new_train.relative_to(tmp_path).as_posix()
    )
    write_json(new_index, dataclasses.asdict(relocated))
    if changed:
        settings = json.loads(new_train.read_text())
        write_json(new_train, settings | {"optimizer_setting": "changed"})
    data = Index(world.out)
    try:
        if changed:
            with pytest.raises(ValueError, match="inputs changed"):
                data.validate()
            with pytest.raises(ValueError, match="inputs changed"):
                index.run(relocated, new_index)
        else:
            data.validate()
            assert index.run(relocated, new_index) == expected
        assert (world.out / "run.json").read_bytes() == original
    finally:
        data.close()
