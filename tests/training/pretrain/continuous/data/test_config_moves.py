"""Config relocation and optimizer changes preserve the data index identity."""

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
        data.validate()
        assert index.run(relocated, new_index) == expected
        assert (world.out / "run.json").read_bytes() == original
    finally:
        data.close()


def test_legacy_v2_index_survives_new_k_and_batch_without_rewriting(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    run = index.run(world.cfg, world.path)
    run["mathematics"]["tbptt_states"] = 2
    run["fingerprint"].update(tbptt_states=2, training_config="legacy-config-hash")
    run["populations"]["train"]["M"] = 7
    write_json(world.out / "run.json", run)
    original = {
        path: path.read_bytes() for path in world.out.rglob("*") if path.is_file()
    }
    path = tmp_path / "train.yaml"
    write_json(
        path, json.loads(path.read_text()) | {"tbptt_states": 4, "batch_size": 3}
    )
    data = Index(world.out)
    try:
        data.validate()
        assert index.run(world.cfg, world.path) == run
    finally:
        data.close()
    assert all(path.read_bytes() == contents for path, contents in original.items())
