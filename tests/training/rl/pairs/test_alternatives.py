"""Equation (4): independent nearest-distance oracles over real History shards."""

import json
import math

import torch

from mcpref.training.rl.pairs.alternatives import nearest
from mcpref.training.rl.pairs.history import History
from mcpref.training.rl.pairs.types import StateRef
from training.rl.pairs.helpers import config, from_dict, vertex, write_history


def naive_nearest(history_dir, seed_key, alpha, count):
    found = []
    shards = sorted((history_dir / "000000/shards").glob("*/vertices.json"))
    for path in shards:
        for record in json.loads(path.read_text()):
            key = record["key"]
            distance = math.sqrt(
                sum((a * (s - k)) ** 2 for a, s, k in zip(alpha, seed_key, key))
            )
            found.append((distance, from_dict(record)))
    found.sort(key=lambda item: (item[0], item[1]))
    return [ref for _, ref in found[:count]]


def test_nearest_matches_naive_float64_disk_loop_with_cross_shard_ties(
    tmp_path, monkeypatch
):
    refs = [StateRef("old", i) for i in range(9)]
    keys = [
        [3.0, 4.0],
        [0.0, 5.0],
        [1.25, -2.5],
        [-3.0, -4.0],
        [5.0, 0.0],
        [0.125, 0.25],
        [-4.0, 3.0],
        [2.0, -1.0],
        [0.0, -5.0],
    ]
    cfg = config(tmp_path, history_per_shard=2)
    write_history(cfg, 0, [vertex(ref, key) for ref, key in zip(refs, keys)])
    history = History.load(tmp_path / "history", 2)
    seed = StateRef("current", 0)
    seed_key = [0.0, 0.0]
    history.stage_seed(seed, seed_key)
    alpha = torch.tensor([1.0, 1.0], dtype=torch.float64)
    norm_calls = 0
    real_norm = torch.linalg.vector_norm

    def counted_norm(*args, **kwargs):
        nonlocal norm_calls
        norm_calls += 1
        return real_norm(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", counted_norm)
    actual = nearest(seed, history, alpha, 7)
    expected = naive_nearest(tmp_path / "history", seed_key, alpha.tolist(), 7)

    assert norm_calls == 5

    assert actual == expected
    assert (
        len([ref for ref in actual if math.dist(keys[ref.index], seed_key) == 5]) == 4
    )


def test_nearest_preserves_staged_insertions_key_replacement_and_removals(tmp_path):
    refs = [StateRef("old", i) for i in range(4)]
    cfg = config(tmp_path, history_per_shard=2)
    write_history(
        cfg,
        0,
        [vertex(ref, [float(i + 1)]) for i, ref in enumerate(refs)],
    )
    history = History.load(tmp_path / "history", 2)
    seed = StateRef("z-current", 0)
    staged = StateRef("staged", 0)
    history.stage_seed(seed, [0.0])
    history.stage_seed(staged, [99.0])
    history.replace_key(staged, [0.5])
    history.cleanup(capacity=5)

    assert nearest(seed, history, torch.tensor([1.0]), 4) == [
        staged,
        refs[1],
        refs[2],
        refs[3],
    ]
