"""(3) of specs/vpt-gui-frames.md: `plan` with a fake `fetch` against a transcription of (3),
and its statistics over many seeds."""

import collections
import dataclasses
import math

import pytest
from vpt_world import (
    BAD_JSONL,
    FIELDS,
    SUBSETS,
    Ep,
    eligible,
    make_world,
    options,
    reference_plan,
    same_options,
)

from mcpref.vpt.gui_frames import GuiFramesConfig, plan

A6 = Ep("data/6.12/a6.mp4", gui=".##.##")
D10 = Ep("data/10.0/d10.mp4", gui=".##")
MIXED = {  # 8xx: empty index; 6xx and 10xx list exact repeats; 7xx: one stem, two versions
    "6xx": [Ep("data/6.9/old6.mp4", reason="version"), A6,
            Ep("data/6.13/m6.mp4", reason="missing", options=None), A6,
            Ep("data/6.10/v6.mp4", reason="version")],
    "7xx": [Ep("data/7.1/n7.mp4", gui="#.#.#.", reason="no_gui"),
            Ep("data/7.2/o7.mp4", reason="options", options=options(guiScale=3)),
            Ep("data/7.1/twin.mp4", reason="missing", options=None, jsonl=None),
            Ep("data/7.2/b7.mp4", gui="..#####"), Ep("data/7.2/twin.mp4", gui="##.##")],
    "9xx": [Ep("data/9.0/e9.mp4", reason="empty", jsonl=b""),
            Ep("data/9.1/x9.mp4", reason="malformed", jsonl=BAD_JSONL),
            Ep("data/9.1/c9.mp4", gui="###")],
    "10xx": [D10, Ep("data/10.0/m10.mp4", reason="missing", jsonl=None),
             Ep("data/10.0/f10.mp4", gui="..########.."), D10],
}  # fmt: skip
SEEDS = range(25)


def config(world, **changes) -> GuiFramesConfig:
    fields = FIELDS | {
        "index_urls": world.index_urls,
        "ffmpeg": "ffmpeg",
        "out_dir": "o",
    }
    return GuiFramesConfig(**(fields | changes))


def as_fields(planned) -> list[dict]:
    keys = ("id", "subset", "relpath", "version", "frame")
    return [
        {k: getattr(p, k) for k in keys} | {"lines": tuple(p.lines)} for p in planned
    ]


def assert_plan(world, cfg) -> None:
    frames, records = plan(cfg, world.fetch)
    exp_frames, exp_records = reference_plan(
        world, cfg.n_frames, cfg.per_episode, cfg.seed
    )
    assert as_fields(frames) == exp_frames
    assert [(r.subset, r.relpath, r.reason) for r in records] == [
        e[:3] for e in exp_records
    ]
    assert all(same_options(r.options, e[3]) for r, e in zip(records, exp_records))


@pytest.mark.parametrize("n", [1, 4, 9, 100])
@pytest.mark.parametrize("m", [1, 3])
def test_plan_matches_transcription_eq3(n, m):
    """Subset choice, draws within a subset until valid, pool exhaustion, the cut at N."""
    world = make_world(MIXED)
    for seed in SEEDS:
        assert_plan(world, config(world, n_frames=n, per_episode=m, seed=seed))


def test_plan_draws_subsets_in_config_order():
    world = make_world(MIXED, order=("10xx", "8xx", "7xx", "6xx", "9xx"))
    for seed in SEEDS:
        assert_plan(world, config(world, n_frames=7, per_episode=2, seed=seed))


def test_plan_exhausts_every_pool_once_without_repeats():
    world = make_world(MIXED)
    valid = [e for es in MIXED.values() for e in set(es) if e.reason is None]
    for seed in SEEDS:
        frames, records = plan(
            config(world, n_frames=1000, per_episode=2, seed=seed), world.fetch
        )
        drawn = [(r.subset, r.relpath) for r in records]
        assert sorted(drawn) == sorted(world.eps)
        assert len(frames) == sum(min(2, len(eligible(e.gui))) for e in valid)


@pytest.mark.parametrize("n", range(1, 14))
def test_plan_stops_at_n_frames(n):
    """len = min(N, all available); the last drawn episode is the one completing N."""
    world = make_world(MIXED)
    for seed in SEEDS:
        frames, records = plan(
            config(world, n_frames=n, per_episode=3, seed=seed), world.fetch
        )
        assert len(frames) == min(n, 13)
        assert n > 13 or records[-1].reason is None


def test_plan_frames_distinct_eligible_and_paired_with_lines():
    world = make_world(MIXED)
    for seed in SEEDS:
        frames, _ = plan(
            config(world, n_frames=100, per_episode=3, seed=seed), world.fetch
        )
        by_episode = collections.defaultdict(list)
        for p in frames:
            by_episode[p.subset, p.relpath].append(p.frame)
        for key, got in by_episode.items():
            f_e = eligible(world.eps[key].gui)
            assert len(set(got)) == len(got) == min(3, len(f_e)) and set(got) <= set(
                f_e
            )
        assert not [
            u for u in world.fetched if u.endswith(".mp4")
        ]  # (2): .mp4 not checked


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


def seeded_plans(world, k: int, **changes):
    cfg = config(world, **changes)
    return (plan(dataclasses.replace(cfg, seed=seed), world.fetch) for seed in range(k))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_subsets_drawn_uniformly_whatever_their_sizes():
    """Pools of 3..48 episodes, 3 draws: every draw picks each subset with p = 1/5."""
    sizes = dict(zip(SUBSETS, (3, 6, 12, 24, 48)))
    eps = {
        s: [Ep(f"data/7.1/{s}-{k}.mp4", gui="##") for k in range(n)]
        for s, n in sizes.items()
    }
    counts = collections.Counter()
    for _, records in seeded_plans(make_world(eps), 4000, n_frames=3, per_episode=1):
        counts.update(r.subset for r in records)
    counts_within(counts, dict.fromkeys(SUBSETS, 1 / 5), 12000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_episode_uniform_within_pool_after_repeats_removed():
    a, b, c, d = (Ep(f"data/7.1/{x}.mp4", gui="##") for x in "abcd")
    world = make_world({"7xx": [a, b, a, c, d, a]})
    counts = collections.Counter()
    for _, records in seeded_plans(world, 4000, n_frames=1, per_episode=1):
        counts[records[0].relpath] += 1
    counts_within(counts, {e.relpath: 1 / 4 for e in (a, b, c, d)}, 4000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_planned_episode_uniform_among_valid_ones():
    gone = [
        Ep(f"data/7.1/gone{k}.mp4", reason="missing", options=None) for k in range(3)
    ]
    valid = [Ep(f"data/7.1/ok{k}.mp4", gui="##") for k in range(3)]
    world = make_world(
        {"7xx": [gone[0], valid[0], gone[1], valid[1], valid[2], gone[2]]}
    )
    counts = collections.Counter()
    for frames, _ in seeded_plans(world, 4000, n_frames=1, per_episode=1):
        counts[frames[0].relpath] += 1
    counts_within(counts, {e.relpath: 1 / 3 for e in valid}, 4000)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_frames_uniform_among_eligible_in_draw_order():
    """F_e = [1..5], m = 2: each ordered pair of distinct frames has p = 1/20."""
    world = make_world({"9xx": [Ep("data/9.0/g.mp4", gui="#" * 6)]})
    counts = collections.Counter()
    for frames, _ in seeded_plans(world, 4000, n_frames=10, per_episode=2):
        counts[tuple(p.frame for p in frames)] += 1
    pairs = [(i, j) for i in range(1, 6) for j in range(1, 6) if i != j]
    counts_within(counts, dict.fromkeys(pairs, 1 / 20), 4000)
