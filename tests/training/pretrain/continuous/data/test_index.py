"""Hand-counted populations, suffix collisions and unchanged edge orientation."""

import dataclasses
import json

import pytest
from training.pretrain.continuous.data.world import make_world

from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.timeline import identity
from mcpref.infrastructure.runs.files import clip_fields, fingerprint, write_json
from mcpref.infrastructure.shards.store import shard_path
from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index


def test_populations_original_edges_aliases_and_suffix_collisions(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    run = index.run(world.cfg, world.path)
    assert run["populations"] == {
        "train": {
            "N_phi": 7,
            "N_pi": 7,
            "N_P": 4,
            "M": 5,
            "observations": 13,
            "fragments": 2,
        },
        "validation": {
            "N_phi": 1,
            "N_pi": 1,
            "N_P": 0,
            "M": 1,
            "observations": 2,
            "fragments": 1,
        },
    }
    data = Index(world.out)
    try:
        data.validate()
        a = identity(world.histories["a"])
        completed = list(data.endpoints("f0", "f0:2"))
        assert len(completed) == 1
        assert completed[0]["observation_ids"] == ["f0:1", "f0:2"]
        assert completed[0]["original_ids"] == ["a", "alias_a"]
        assert completed[0]["pending_edges"] == 4
        edges = list(data.edges(a))
        assert [row["id"] for row in edges] == [f"old:{i}" for i in range(4)]
        assert edges[1]["a"] == identity(world.histories["b"])
        assert edges[1]["b"] == a
        assert [row["labels"]["p"] for row in edges] == [0, 0.25, 0.5, 0.75]
        assert list(data.excluded_pairs()) == [
            {
                "id": "old:4",
                "a": "a",
                "b": "short_a",
                "labels": {"inc": 0, "p": 1},
                "reason": "collapsed_pair",
            }
        ]
    finally:
        data.close()


@pytest.mark.parametrize("change", ["history", "K", "mask"])
def test_mathematical_inputs_reject_reusing_an_index(tmp_path, monkeypatch, change):
    world = make_world(tmp_path, monkeypatch)
    index.run(world.cfg, world.path)
    if change == "history":
        write_json(tmp_path / "preference.yaml", {"history": 1})
    elif change == "K":
        train = json.loads((tmp_path / "train.yaml").read_text())
        write_json(tmp_path / "train.yaml", train | {"tbptt_states": 4})
    else:
        monkeypatch.setattr(
            index,
            "load_masks",
            lambda _: dataclasses.replace(world.masks, pitch_limit=89),
        )
    with pytest.raises(ValueError, match="inputs changed"):
        index.run(world.cfg, world.path)


def test_interrupted_index_reuses_complete_descriptor_shards(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    save = index.save

    def interrupt(out, k, items):
        save(out, k, items)
        raise RuntimeError("interrupted after publication")

    monkeypatch.setattr(index, "save", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        index.run(world.cfg, world.path)
    first = (world.out / "shards/000000/fragments.json").read_bytes()
    monkeypatch.setattr(index, "save", save)
    population = index.metadata.population

    def only_remaining(root, fragment, masks, K):
        assert fragment["id"] != "f0"
        return population(root, fragment, masks, K)

    monkeypatch.setattr(index.metadata, "population", only_remaining)
    run = index.run(world.cfg, world.path)
    assert run["populations"]["train"]["N_phi"] == 7
    assert (world.out / "shards/000000/fragments.json").read_bytes() == first
    assert index.run(world.cfg, world.path) == run


def test_additional_shared_labels_are_indexed_without_replacing_old_edges(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    pool = tmp_path / "pool"
    pool_run = {
        "complete": True,
        "shared_dataset": "dataset",
        "shared_fingerprint": {"dataset": digest(world.root / "run.json")},
    }
    write_json(pool / "run.json", pool_run)
    write_json(pool / "shards/000000/records.json", [{"clips": ["x", "y"]}])
    refs = []
    for alias in ("a", "b"):
        ids = world.histories[alias][-2:]
        key = identity(ids)
        write_json(
            pool / "shards/000000/clips" / key / "clip.json",
            {
                "id": key,
                "dataset": "dataset",
                "fragment_id": ids[0][:2],
                "observation_ids": ids,
                "frames": [world.positions[v] for v in ids],
                "classical_valid": True,
            },
        )
        refs.append({"shard": 0, "id": key, "episode": ids[0][:2]})
    write_json(
        tmp_path / "labels/run.json",
        {
            "complete": True,
            "pair_clips": fingerprint(
                "pool", "records.json", clip_fields(pool_run), lambda: None
            ),
        },
    )
    write_json(
        tmp_path / "labels/shards/000000/pairs.json",
        [{"a": refs[1], "b": refs[0], "labels": {"inc": 0, "p": 0.125}}],
    )
    cfg = dataclasses.replace(
        world.cfg, pair_clips_dir="pool", pair_labels_dir="labels"
    )
    write_json(world.path, dataclasses.asdict(cfg))
    run = index.run(cfg, world.path)
    assert run["populations"]["train"]["N_P"] == 5
    data = Index(world.out)
    try:
        edge = next(data.edges(refs[1]["id"]))
        assert edge["a"] == refs[1]["id"] and edge["b"] == refs[0]["id"]
        assert edge["labels"] == {"inc": 0, "p": 0.125}
    finally:
        data.close()


def test_changed_split_cannot_separate_original_edge(tmp_path, monkeypatch):
    world = make_world(tmp_path, monkeypatch)
    split = json.loads((world.root / "split.json").read_text())
    split["fragment_parts"]["f1"] = split["groups"]["w1"] = "validation"
    write_json(world.root / "split.json", split)
    with pytest.raises(ValueError, match="crosses the frozen split"):
        index.run(world.cfg, world.path)


def test_metadata_changed_without_its_source_checksum_is_rejected(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    path = shard_path(world.root / "shards", 0) / "intervals.jsonl"
    path.write_text(path.read_text().replace("forward", "turn_up"))
    with pytest.raises(ValueError, match="checksum"):
        index.run(world.cfg, world.path)
