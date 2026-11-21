"""Independent population, boundary and durable-recovery regression oracles."""

import json
from types import SimpleNamespace

import pytest
from datasets.vpt.action_lines import episode

from mcpref.datasets.continuous import recovery
from mcpref.datasets.continuous.boundaries import blocks
from mcpref.datasets.continuous.rows import projection
from mcpref.datasets.continuous.views.index import population


def mapped(name, valid=True, kind="preference", canonical=None):
    return {
        "status": "mapped",
        "kind": kind,
        "canonical_history_id": canonical or name,
        "classical_valid": valid,
        "classical_reasons": [] if valid else ["idle"],
        "clip_ref": name,
        "world_group": "world",
    }


def test_classical_rebuilds_own_union_after_pair_filtering():
    mappings = {
        "u": mapped("u", kind="ordinary"),
        "bad": mapped("bad", False),
        "orphan": mapped("orphan"),
        "a": mapped("a"),
        "b": mapped("b"),
        "alias": mapped("a", canonical="a"),
    }
    pairs = [
        {"id": "drop", "a": "orphan", "b": "bad", "labels": {"x": 1}},
        {"id": "keep", "a": "a", "b": "b", "labels": {"x": 2}},
        {"id": "repeat", "a": "alias", "b": "b", "labels": {"x": 3}},
    ]
    entries, excluded = population(mappings, pairs, {"world": "train"})
    assert [item["clip"] for item, _ in entries if "clip" in item] == ["u", "a", "b"]
    assert [item["labels"] for item, _ in entries if "a" in item] == [
        {"x": 2},
        {"x": 3},
    ]
    assert [item["id"] for item in excluded] == ["drop"]


@pytest.mark.parametrize(
    "gap,start,tick,expected",
    [
        (999, False, 2, 1),
        (1000, False, 2, 2),
        (-1, False, 2, 2),
        (1, True, 2, 2),
        (1, False, 0, 2),
    ],
)
def test_join_boundaries(gap, start, tick, expected):
    rows = [
        {"milli": 1000, "serverTick": 1, "starts_session": False},
        {"milli": 1000 + gap, "serverTick": tick, "starts_session": start},
    ]
    sources = [{"reason": None, "offset": i, "rows": 1} for i in range(2)]
    assert len(list(blocks(sources, rows, 1000))) == expected


def test_invalid_source_breaks_even_when_offsets_touch():
    rows = [
        {"milli": 0, "serverTick": 0, "starts_session": False},
        {"milli": 10, "serverTick": 1, "starts_session": False},
    ]
    sources = [
        {"reason": None, "offset": 0, "rows": 1},
        {"reason": "missing"},
        {"reason": None, "offset": 1, "rows": 1},
    ]
    assert list(blocks(sources, rows, 1000)) == [
        [sources[0]],
        [sources[1]],
        [sources[2]],
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"yaw": "0.0"},
        {"hotbar": 9},
        {"keyboard": {"keys": [17], "newKeys": []}},
        {"inventory": [{"type": "stone", "quantity": float("nan")}]},
    ],
)
def test_malformed_source_rejected_before_evidence(change):
    row = episode("w w")[0] | change
    with pytest.raises((TypeError, ValueError)):
        projection(row, 0)


def test_committed_fragment_cannot_resume_without_payload(tmp_path):
    directory = tmp_path / "fragments/000000"
    directory.mkdir(parents=True)
    summary = {"first_shard": 0, "next_shard": 1, "sources": [], "manifest_index": 0}
    (directory / "summary.json").write_text(json.dumps(summary))
    recovery.seal(directory)
    (tmp_path / "shards/000000").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        recovery.recover(SimpleNamespace(), tmp_path, 0)


def test_committed_payload_corruption_is_rejected(tmp_path):
    directory = tmp_path / "fragments/000000"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text("{}")
    recovery.seal(directory)
    (directory / "summary.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="checksum"):
        recovery.verify_files(directory)
