"""Statistics of (3) of specs/vpt-gui-frames.md over many seeds (slow tier): subset shares,
uniformity among episodes and among eligible frames."""

import collections
import dataclasses
import math
import os

import pytest
from vpt_checks import plan_config, read_shards
from vpt_world import SUBSETS, Ep, make_world

from mcpref.vpt.gui_frames import plan


def counts_within(counts: collections.Counter, p: dict, trials: int) -> None:
    """Each outcome's count within 5 sigma of its exact expectation."""
    assert set(counts) <= set(p)
    for outcome, q in p.items():
        sigma = math.sqrt(trials * q * (1 - q))
        assert abs(counts[outcome] - trials * q) <= 5 * sigma, (
            outcome,
            counts,
            trials * q,
        )


def seeded_plans(world, k: int, tmp_path, monkeypatch, **changes):
    """Shards of the plans of seeds 0, ..., k - 1, one shard each; `os.fsync` is a no-op
    (durability is not under test here)."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    cfg = plan_config(world, plan_shard_draws=10**6, **changes)
    for seed in range(k):
        plan(dataclasses.replace(cfg, seed=seed), world.fetch, tmp_path / str(seed))
        yield read_shards(tmp_path / str(seed))[0]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_subsets_drawn_uniformly_whatever_their_sizes(tmp_path, monkeypatch):
    """Pools of 3..48 episodes, 3 draws: every draw picks each subset with p = 1/5."""
    sizes = dict(zip(SUBSETS, (3, 6, 12, 24, 48)))
    eps = {
        s: [Ep(f"data/7.1/{s}-{k}.mp4", gui="##") for k in range(n)]
        for s, n in sizes.items()
    }
    counts = collections.Counter()
    plans = seeded_plans(make_world(eps), 4000, tmp_path, monkeypatch, n_frames=3,
                         per_episode=1)  # fmt: skip
    for shard in plans:
        counts.update(r["subset"] for r in shard["episodes"])
    counts_within(counts, dict.fromkeys(SUBSETS, 1 / 5), 12000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_episode_uniform_within_pool_after_repeats_removed(tmp_path, monkeypatch):
    a, b, c, d = (Ep(f"data/7.1/{x}.mp4", gui="##") for x in "abcd")
    world = make_world({"7xx": [a, b, a, c, d, a]})
    counts = collections.Counter()
    plans = seeded_plans(world, 4000, tmp_path, monkeypatch, n_frames=1, per_episode=1)
    for shard in plans:
        counts[shard["episodes"][0]["relpath"]] += 1
    counts_within(counts, {e.relpath: 1 / 4 for e in (a, b, c, d)}, 4000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_planned_episode_uniform_among_valid_ones(tmp_path, monkeypatch):
    gone = [
        Ep(f"data/7.1/gone{k}.mp4", reason="missing", options=None) for k in range(3)
    ]
    valid = [Ep(f"data/7.1/ok{k}.mp4", gui="##") for k in range(3)]
    world = make_world(
        {"7xx": [gone[0], valid[0], gone[1], valid[1], valid[2], gone[2]]}
    )
    counts = collections.Counter()
    plans = seeded_plans(world, 4000, tmp_path, monkeypatch, n_frames=1, per_episode=1)
    for shard in plans:
        counts[shard["frames"][0]["relpath"]] += 1
    counts_within(counts, {e.relpath: 1 / 3 for e in valid}, 4000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_frames_uniform_among_eligible_in_draw_order(tmp_path, monkeypatch):
    """F_e = [1..5], m = 2: each ordered pair of distinct frames has p = 1/20."""
    world = make_world({"9xx": [Ep("data/9.0/g.mp4", gui="#" * 6)]})
    counts = collections.Counter()
    plans = seeded_plans(world, 4000, tmp_path, monkeypatch, n_frames=10, per_episode=2)
    for shard in plans:
        counts[tuple(p["frame"] for p in shard["frames"])] += 1
    pairs = [(i, j) for i in range(1, 6) for j in range(1, 6) if i != j]
    counts_within(counts, dict.fromkeys(pairs, 1 / 20), 4000)
