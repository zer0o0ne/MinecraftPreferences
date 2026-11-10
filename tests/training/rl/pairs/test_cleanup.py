"""Equation (10): cleanup scores, ties, repeated removal, and symmetric deletion."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mcpref.infrastructure.runs.files import json_bytes
from mcpref.training.rl.pairs.cycle import publish_memory, sample_pairs
from mcpref.training.rl.pairs.history import History
from mcpref.training.rl.pairs.types import (
    Comparison,
    CurrentComparison,
    StateRef,
    Vertex,
    read_vertex,
    ref_dict,
    state_ref,
    vertex_json,
)
from training.rl.pairs.helpers import (
    Encoder,
    Preference,
    comparison,
    config,
    from_dict,
    vertex,
    vertices,
    write_history,
)


def linked_records(refs):
    a, b, c, d = refs
    return [
        vertex(
            a,
            [0.0],
            comparisons=[comparison(a, b, 0.8, 0.1, 1), comparison(a, c, 0.7, 0.2, 4)],
        ),
        vertex(b, [1.0], comparisons=[comparison(b, a, 0.2, 0.1, 1)]),
        vertex(c, [2.0], comparisons=[comparison(c, a, 0.3, 0.2, 4)]),
        vertex(d, [3.0]),
    ]


def publish_empty(cfg, cycle):
    sample_pairs(
        cfg,
        cycle,
        [],
        Encoder({}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    publish_memory(cfg, cycle, Encoder({}))


def test_cleanup_repeated_removal_reference_deletion_and_count_subtraction(tmp_path):
    refs = [StateRef("old", i) for i in range(4)]
    a, _, c, _ = refs
    cfg = config(tmp_path, sample_states=0, capacity=2)
    write_history(cfg, 0, linked_records(refs))
    publish_empty(cfg, 1)
    stored = {from_dict(v): v for v in vertices(tmp_path / "history", 1)}
    assert set(stored) == {a, c}
    assert stored[a]["comparisons"] == [comparison(a, c, 0.7, 0.2, 4)]
    assert stored[c]["comparisons"] == [comparison(c, a, 0.3, 0.2, 4)]
    assert stored[a]["unused"] == stored[c]["unused"] == 1


@pytest.mark.parametrize("unused", [5, 6], ids=["stable-id-tie", "larger-unused"])
def test_cleanup_score_tie_uses_larger_unused_then_stable_id(tmp_path, unused):
    refs = [StateRef("old", i) for i in range(4)]
    records = [vertex(refs[0], [0.0], comparisons=[comparison(refs[0], refs[1], 0.5, 0.0, 2)]),
               vertex(refs[1], [1.0], comparisons=[comparison(refs[1], refs[0], 0.5, 0.0, 2)]),
               vertex(refs[2], [2.0], unused=unused), vertex(refs[3], [3.0], unused=5)]  # fmt: skip
    cfg = config(tmp_path, sample_states=0, capacity=3)
    write_history(cfg, 0, records)
    publish_empty(cfg, 1)
    assert {from_dict(v) for v in vertices(tmp_path / "history", 1)} == {
        refs[0],
        refs[1],
        refs[3],
    }


def test_exact_capacity_boundary_removes_nothing(tmp_path):
    refs = [StateRef("old", i) for i in range(2)]
    cfg = config(tmp_path, sample_states=0, capacity=2)
    write_history(cfg, 0, [vertex(refs[0], [0.0], unused=2),
                           vertex(refs[1], [1.0], unused=4)])  # fmt: skip
    publish_empty(cfg, 1)
    stored = vertices(tmp_path / "history", 1)
    assert [from_dict(v) for v in stored] == refs
    assert [v["unused"] for v in stored] == [3, 5]


def full_rescan(records, capacity):
    remaining = [read_vertex(v) if isinstance(v, dict) else v for v in records]
    removed = []
    while len(remaining) > capacity:
        victim = min(
            remaining,
            key=lambda v: (math.sqrt(v.n / (v.unused + 1)), -v.unused, v.ref),
        )
        removed.append(victim.ref)
        remaining = [
            Vertex(
                v.ref,
                v.key,
                v.unused,
                tuple(c for c in v.comparisons if c.ref != victim.ref),
            )
            for v in remaining
            if v.ref != victim.ref
        ]
    return removed, remaining


def generation_files(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def test_cleanup_matches_full_rescan_with_staging_and_resume(tmp_path):
    a, b, c, d, x, fresh = [StateRef("old", i) for i in range(6)]
    records = [
        vertex(a, [0.0], comparisons=[comparison(a, x, 0.5, 0.0, 1)]),
        vertex(b, [1.0], comparisons=[comparison(b, x, 0.5, 0.0, 1)]),
        vertex(c, [2.0], unused=3, comparisons=[comparison(c, d, 0.5, 0.0, 1)]),
        vertex(d, [3.0], comparisons=[comparison(d, c, 0.5, 0.0, 1)]),
        vertex(
            x,
            [4.0],
            unused=10,
            comparisons=[
                comparison(x, a, 0.5, 0.0, 1),
                comparison(x, b, 0.5, 0.0, 1),
            ],
        ),
    ]
    pending = [
        Vertex(
            a,
            (0.0,),
            0,
            (
                Comparison(x, 0.5, 0.0, 1),
                Comparison(fresh, 0.25, 0.2, 1),
            ),
        ),
        Vertex(b, (1.0,), 1, (Comparison(x, 0.5, 0.0, 1),)),
        Vertex(c, (2.0,), 4, (Comparison(d, 0.5, 0.0, 1),)),
        Vertex(d, (3.0,), 1, (Comparison(c, 0.5, 0.0, 1),)),
        Vertex(
            x,
            (4.0,),
            11,
            (
                Comparison(a, 0.5, 0.0, 1),
                Comparison(b, 0.5, 0.0, 1),
            ),
        ),
        Vertex(fresh, (9.0,), 0, (Comparison(a, 0.75, 0.2, 1),)),
    ]
    expected_order, expected = full_rescan(pending, capacity=4)
    assert expected_order == [x, b]

    def prepare(root, restored=()):
        cfg = config(root, history_per_shard=2)
        write_history(cfg, 0, records)
        history = History.load(Path(cfg.history_dir), 2, {"history_per_shard": 2})
        history.stage_seed(fresh, [5.0])
        history.replace_key(fresh, [9.0])
        history.stage_comparison(CurrentComparison(a, fresh, 0.25, 0.2))
        history.age_unused()
        checkpoint = root / "cleanup.json"
        if restored:
            checkpoint.write_bytes(
                json_bytes({"removed": [ref_dict(r) for r in restored]})
            )
        return history, checkpoint

    history, checkpoint = prepare(tmp_path / "baseline")
    history.cleanup(4, checkpoint)
    published = history.publish(1)
    resumed, resumed_checkpoint = prepare(tmp_path / "resumed", expected_order[:1])
    resumed.cleanup(4, resumed_checkpoint)
    resumed_published = resumed.publish(1)

    saved = [state_ref(r) for r in json.loads(checkpoint.read_text())["removed"]]
    assert saved == expected_order
    assert resumed_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert generation_files(resumed_published) == generation_files(published)
    for start in range(0, len(expected), 2):
        shard = published / "shards" / f"{start // 2:06d}" / "vertices.json"
        expected_json = [vertex_json(v) for v in expected[start : start + 2]]
        assert shard.read_bytes() == json_bytes(expected_json)
