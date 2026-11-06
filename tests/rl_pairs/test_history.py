"""Equations (7), (11), (12): comparisons, fixed Q_c, and deferred History publication."""

import numpy as np
import pytest
import torch

from mcpref.rl_pairs.cycle import publish_memory, sample_pairs
from mcpref.rl_pairs.types import StateRef

from .helpers import (
    Encoder,
    Preference,
    config,
    from_dict,
    items,
    pair,
    vertex,
    vertices,
    work,
    write_history,
)


def expected_unique(results):
    expected, seen = [], set()
    for result in results:
        seed = from_dict(result["seed"])
        for value in result["alternatives"]:
            alternative = from_dict(value)
            key = tuple(sorted((seed, alternative)))
            if key not in seen:
                seen.add(key)
                expected.append((seed, alternative))
    return expected


def test_current_pairs_global_dedup_orientation_and_seed_only_vertices(tmp_path):
    cfg = config(tmp_path, sample_states=3, temporal_per_state=3,
                 history_per_shard=2)  # fmt: skip
    states = [StateRef("e", i) for i in range(5)]
    values = {r: (float(i + 1),) for i, r in enumerate(states)}
    preference = Preference(default=(0.75, 0.0)).train()
    out = sample_pairs(
        cfg,
        0,
        states,
        Encoder(values),
        preference,
        torch.tensor([1.0]),
        np.random.default_rng(23),
    )
    results = work(out)
    seeds = [from_dict(result["seed"]) for result in results]
    expected = expected_unique(results)
    assert [(from_dict(c["a"]), from_dict(c["b"]))
            for result in results for c in result["comparisons"]] == expected  # fmt: skip
    assert [pair(item) for item in items(out)] == [
        ((a.episode, a.index), (b.episode, b.index)) for a, b in expected
    ]
    assert {ref for item in items(out) for ref in pair(item)} <= {
        (r.episode, r.index) for r in seeds
    }
    assert vertices(tmp_path / "history", 0) == []
    publish_memory(cfg, 0, Encoder(values))
    stored = vertices(tmp_path / "history", 0)
    assert [from_dict(v) for v in stored] == sorted(seeds)
    assert all(len(v["comparisons"]) == 2 for v in stored)
    assert preference.training and preference.dummy.grad is None


class TwoCallPreference(Preference):
    def __init__(self, current):
        super().__init__()
        self.current = current

    def forward(self, a, b):
        self.default = (0.5, 0.0) if not self.calls else self.current
        return super().forward(a, b)


def test_current_cycle_updates_symmetric_records_and_preserves_old_keys(tmp_path):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1)
    states = [StateRef("e", 0), StateRef("e", 1)]
    first_encode = Encoder({states[0]: (1.0, 0.0), states[1]: (2.0, 0.0)})
    first = TwoCallPreference((0.75, 0.1))
    sample_pairs(
        cfg,
        0,
        states,
        first_encode,
        first,
        torch.tensor([1.0, 1.0]),
        np.random.default_rng(5),
    )
    publish_memory(cfg, 0, first_encode)
    before = {from_dict(v): v["key"] for v in vertices(tmp_path / "history", 0)}
    states = [StateRef("e1", 0), StateRef("e1", 1)]
    second_encode = Encoder({states[0]: (1.0, 9.0), states[1]: (2.0, 9.0)})
    second = TwoCallPreference((0.25, 0.2)).train()
    out = sample_pairs(
        cfg,
        1,
        states,
        second_encode,
        second,
        torch.tensor([1.0, 1.0]),
        np.random.default_rng(5),
    )
    raw = next(c for result in work(out) for c in result["comparisons"])
    a, b = from_dict(raw["a"]), from_dict(raw["b"])
    assert vertices(tmp_path / "history", 1) == []
    publish_memory(cfg, 1, second_encode)
    stored = {from_dict(v): v for v in vertices(tmp_path / "history", 1)}
    forward = next(c for c in stored[a]["comparisons"] if from_dict(c) == b)
    reverse = next(c for c in stored[b]["comparisons"] if from_dict(c) == a)
    assert from_dict(forward) == b
    assert from_dict(reverse) == a
    assert forward["p"] == pytest.approx(0.25)
    assert reverse["p"] == pytest.approx(0.75)
    assert forward["inc"] == reverse["inc"] == pytest.approx(0.2)
    assert forward["n"] == reverse["n"] == 1
    assert {r: v["key"] for r, v in stored.items()} == before | {
        r: [float(r.index + 1), 9.0] for r in states
    }
    assert second.training and second.dummy.grad is None


def test_usage_aging_depends_on_actual_comparison(tmp_path):
    cfg = config(tmp_path, sample_states=1, temporal_per_state=0, nearest_per_state=1)
    old_a, old_b, seed = StateRef("old", 0), StateRef("old", 1), StateRef("new", 0)
    write_history(
        cfg, 0, [vertex(old_a, [0.0], unused=4), vertex(old_b, [50.0], unused=7)]
    )
    out = sample_pairs(
        cfg,
        1,
        [seed],
        Encoder({seed: (0.0,)}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    assert work(out)[0]["comparisons"]
    assert vertices(tmp_path / "history", 1) == []
    publish_memory(cfg, 1, Encoder({seed: (3.0,)}))
    stored = {from_dict(v): v for v in vertices(tmp_path / "history", 1)}
    assert stored[old_a]["unused"] == 0
    assert stored[old_b]["unused"] == 8
    assert stored[seed]["unused"] == 0


def test_empty_candidates_and_empty_preference_dataset_are_valid(tmp_path):
    cfg = config(tmp_path, sample_states=1, temporal_per_state=1)
    seed = StateRef("e", 0)
    preference = Preference().train()
    out = sample_pairs(
        cfg,
        0,
        [seed],
        Encoder({seed: (0.0,)}),
        preference,
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    assert items(out) == [] and work(out)[0]["comparisons"] == []
    assert vertices(tmp_path / "history", 0) == []
    publish_memory(cfg, 0, Encoder({seed: (0.0,)}))
    assert len(vertices(tmp_path / "history", 0)) == 1
    assert preference.training and preference.dummy.grad is None


def test_fixed_snapshot_survives_network_change_and_later_eviction(tmp_path):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1, capacity=1)
    first_states = [StateRef("first", 0), StateRef("first", 1)]
    first_encode = Encoder({r: (float(r.index),) for r in first_states})
    first_out = sample_pairs(cfg, 0, first_states, first_encode,
                             Preference(default=(0.8, 0.0)), torch.tensor([1.0]),
                             np.random.default_rng(4))  # fmt: skip
    before = [item.copy() for item in items(first_out)]
    assert len(before) == 1 and vertices(tmp_path / "history", 0) == []
    publish_memory(cfg, 0, first_encode)
    assert len(vertices(tmp_path / "history", 0)) == 1
    second_states = [StateRef("second", 0), StateRef("second", 1)]
    second_encode = Encoder({r: (99.0,) for r in second_states})
    sample_pairs(
        cfg,
        1,
        second_states,
        second_encode,
        Preference(default=(0.1, 1.0)),
        torch.tensor([1.0]),
        np.random.default_rng(4),
    )
    assert items(first_out) == before
    publish_memory(cfg, 1, second_encode)
    assert items(first_out) == before
    assert {from_dict(v).episode for v in vertices(tmp_path / "history", 1)} == {
        "second"
    }


def test_publication_reencodes_only_fresh_seeds_and_preserves_old_keys(tmp_path):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=0)
    old = StateRef("old", 0)
    states = [StateRef("new", i) for i in range(4)]
    write_history(cfg, 0, [vertex(old, [7.0, 8.0], unused=3)])
    sample_pairs(
        cfg,
        1,
        states,
        Encoder({r: (1.0, 1.0) for r in states}),
        Preference(),
        torch.tensor([1.0, 1.0]),
        np.random.default_rng(13),
    )
    final = Encoder({r: (10.0 + r.index, 20.0 + r.index) for r in states})
    publish_memory(cfg, 1, final)
    stored = {from_dict(v): v for v in vertices(tmp_path / "history", 1)}
    seeds = [from_dict(result["seed"]) for result in work(tmp_path / "out/000001")]
    assert final.calls == [(ref,) for ref in seeds]
    assert stored[old]["key"] == [7.0, 8.0]
    assert stored[old]["unused"] == 4
    assert {ref for ref in stored if ref.episode == "new"} == set(seeds)
    for ref in seeds:
        assert stored[ref]["key"] == [10.0 + ref.index, 20.0 + ref.index]
