"""Shared index resume and additional-label population contracts."""

import dataclasses
from types import SimpleNamespace

import pytest

from mcpref import config
from mcpref.datasets.continuous.views import index, labels
from mcpref.infrastructure.runs.files import write_json
from mcpref.training.pretrain.classical.data.index import TrainIndexConfig


def test_inherited_shard_size_change_rejects_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "REPO", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    root = tmp_path / "shared"
    write_json(
        root / "run.json",
        {"complete": True, "width": 2, "height": 2, "fingerprint": {}},
    )
    write_json(
        root / "split.json", {"seed": 0, "validation": "0", "groups": {"g": "train"}}
    )
    write_json(tmp_path / "clip/clip.json", {"states": [0, 1, 2], "id": "c"})
    write_json(
        root / "classical-view/000000/items.json",
        {"items": [{"clip": "clip", "part": "train"}]},
    )
    parent = tmp_path / "base.yaml"
    parent.write_text("items_per_shard: 1\n")
    path = tmp_path / "index.yaml"
    path.write_text("extends:\n  - base.yaml\n")
    cfg = TrainIndexConfig("shared", None, None, "index", "0", 0, 1)
    index.build_index(cfg, path, root)
    before = (tmp_path / "index/train/000000/items.json").read_bytes()
    parent.write_text("items_per_shard: 2\n")
    with pytest.raises(RuntimeError, match="config changed"):
        index.build_index(dataclasses.replace(cfg, items_per_shard=2), path, root)
    assert (tmp_path / "index/train/000000/items.json").read_bytes() == before


def test_additional_labels_keep_unique_own_clips_and_repeated_pairs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(labels, "REPO", tmp_path)
    write_json(tmp_path / "old/clip.json", {"id": "same"})
    items = [[{"clip": "old"}], []]
    extra = (
        [("alias", {"id": "same"}, "train"), ("new", {"id": "new"}, "train")],
        [
            ({"a": "alias", "b": "new", "labels": {"x": 1}}, "train"),
            ({"a": "new", "b": "alias", "labels": {"x": 2}}, "train"),
        ],
    )
    labels.attach(items, extra)
    assert items == [
        [
            {"clip": "old"},
            {"clip": "new"},
            {"a": "alias", "b": "new", "labels": {"x": 1}},
            {"a": "new", "b": "alias", "labels": {"x": 2}},
        ],
        [],
    ]


def test_additional_labels_reject_same_path_new_dataset_version(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "REPO", tmp_path)
    write_json(tmp_path / "shared/run.json", {"complete": True, "version": "new"})
    write_json(
        tmp_path / "pool/run.json",
        {
            "complete": True,
            "shared_dataset": "shared",
            "shared_fingerprint": {"dataset": "old-hash"},
        },
    )
    write_json(tmp_path / "labels/run.json", {"complete": True})
    cfg = SimpleNamespace(pair_clips_dir="pool", pair_labels_dir="labels")
    with pytest.raises(ValueError, match="dataset version"):
        labels.extra_labels(cfg, tmp_path / "shared")


def test_new_preference_selection_uses_existing_nonoverlapping_histories():
    import numpy as np

    from mcpref.datasets.continuous.views.pool import select

    candidates = [
        {"id": str(i), "observation_ids": [f"f:{i}", f"f:{i + 1}"]} for i in range(6)
    ]
    occupied = set()
    rng = np.random.Generator(np.random.PCG64(123))
    chosen = select(candidates, 10, rng, occupied)
    assert all(item in candidates for item in chosen)
    assert len({v for c in chosen for v in c["observation_ids"]}) == len(chosen) * 2
    assert select(candidates, 10, rng, occupied) == []


def test_additional_labels_from_frozen_pool_are_loaded_without_relabeling(
    tmp_path, monkeypatch
):
    from mcpref.datasets.continuous.config import digest
    from mcpref.datasets.continuous.timeline import identity
    from mcpref.infrastructure.runs import files

    monkeypatch.setattr(labels, "REPO", tmp_path)
    monkeypatch.setattr(files, "REPO", tmp_path)
    root = tmp_path / "shared"
    write_json(root / "run.json", {"complete": True, "version": "frozen"})
    write_json(root / "split.json", {"fragment_parts": {"fragment": "train"}})
    run = {
        "complete": True,
        "shared_dataset": "shared",
        "shared_fingerprint": {"dataset": digest(root / "run.json")},
    }
    write_json(tmp_path / "pool/run.json", run)
    write_json(tmp_path / "pool/shards/000000/records.json", [])
    names = []
    for observation in ("fragment:0", "fragment:6"):
        name = identity([observation])
        names.append(name)
        write_json(
            tmp_path / "pool/shards/000000/clips" / name / "clip.json",
            {
                "id": name,
                "dataset": "shared",
                "fragment_id": "fragment",
                "classical_valid": True,
                "observation_ids": [observation],
            },
        )
    expected = files.fingerprint(
        "pool", "records.json", files.clip_fields(run), lambda: None
    )
    write_json(tmp_path / "labels/run.json", {"complete": True, "pair_clips": expected})
    payload = {
        "a": {"shard": 0, "id": names[0]},
        "b": {"shard": 0, "id": names[1]},
        "labels": {"incomparable": 0.0, "preference": 0.8},
    }
    write_json(tmp_path / "labels/shards/000000/pairs.json", [payload, payload])
    extra, fingerprint = labels.extra_labels(
        SimpleNamespace(pair_clips_dir="pool", pair_labels_dir="labels"), root
    )
    assert len(extra[0]) == 2
    assert [p["labels"] for p, _ in extra[1]] == [payload["labels"], payload["labels"]]
    assert len(fingerprint) == 64
