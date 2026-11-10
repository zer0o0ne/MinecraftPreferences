"""Atomic typed pairs, one-record shards, arbitrary ids and streaming index."""

import gc
import json
import weakref
from collections.abc import Sequence

import pytest

from mcpref.diagnostics.artifacts import (
    complete_records,
    write_index,
    write_pairs,
    write_record,
)
from mcpref.diagnostics.targets import on_graph

from .world import comparison, config, context, pair


def pending(record_id="graph:non-numeric"):
    record = on_graph(
        [comparison("e0")],
        categories=["comparable"],
        cfg=config(),
        context=context("pair_labels/shards/000003/counts.json", "acyclic-graph"),
    )
    record["id"] = record_id
    assert record["index"] is None
    return record


def test_write_pairs_exact_typed_schema_atomically(tmp_path):
    directory = tmp_path / "reference/000000"
    write_pairs(directory, [pair()])
    path = directory / "pairs.json"
    assert json.loads(path.read_text()) == [
        {
            "micro_batch": 0,
            "pair_in_micro": 0,
            "a": "a",
            "b": "b",
            "labels": {"p": 0.5, "inc": 0},
            "baseline": 3.0,
            "only_phi_post": 2.0,
            "joint_post": 1.0,
            "delta_only_phi": -1.0,
            "delta_joint": -2.0,
        }
    ]
    assert not (directory / "pairs.json.part").exists()


def test_write_record_uses_numeric_index_not_arbitrary_id(tmp_path):
    record = pending()
    write_record(tmp_path, "targets", 0, record)
    path = tmp_path / "targets/000000/metrics.json"
    stored = json.loads(path.read_text())
    assert stored["index"] == 0 and stored["id"] == "graph:non-numeric"
    assert stored["version"] == 1 and stored["kind"] == "targets"
    assert set(stored) == {
        "version",
        "kind",
        "index",
        "id",
        "context",
        "populations",
        "metrics",
        "joined",
    }
    assert not isinstance(stored, list) and record["index"] is None
    assert complete_records(tmp_path, "targets") == 1
    assert not list(tmp_path.rglob("*.part"))


def test_write_record_rejects_noncontiguous_index(tmp_path):
    write_record(tmp_path, "targets", 0, pending())
    with pytest.raises(ValueError):
        write_record(tmp_path, "targets", 2, pending())
    assert complete_records(tmp_path, "targets") == 1


def test_complete_records_is_contiguous_and_index_paths_use_indices(tmp_path):
    first, second = pending("vote:a"), pending("graph:b")
    write_record(tmp_path, "targets", 0, first)
    write_record(tmp_path, "targets", 1, second)
    assert complete_records(tmp_path, "targets") == 2
    write_index(tmp_path, [first | {"index": 0}, second | {"index": 1}])
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["version"] == 1
    assert index["records"] == [
        {
            "kind": "targets",
            "index": i,
            "id": record["id"],
            "path": f"targets/{i:06d}/metrics.json",
            "condition": "train",
            "clusters": record["populations"],
        }
        for i, record in enumerate((first, second))
    ]
    assert not (tmp_path / "index.json.part").exists()
    gap = tmp_path / "targets/000003"
    gap.mkdir()
    (gap / "metrics.json").write_text("{}")
    with pytest.raises(ValueError):
        complete_records(tmp_path, "targets")


class WeakRecord(dict):
    pass


class RecordStream(Sequence):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        raise AssertionError("indexing materializes random access instead of streaming")

    def __iter__(self):
        previous = []
        for i in range(len(self)):
            if len(previous) >= 2:
                gc.collect()
                assert previous[-2]() is None, "write_index retained an older shard"
            record = WeakRecord(pending(f"graph:{i}") | {"index": i})
            previous.append(weakref.ref(record))
            yield record


def test_index_consumes_a_lazy_stream_without_retaining_shards(tmp_path):
    write_index(tmp_path, RecordStream())
    index = json.loads((tmp_path / "index.json").read_text())
    assert [entry["index"] for entry in index["records"]] == list(range(4))
