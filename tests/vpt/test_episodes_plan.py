"""(3) of specs/vpt-gui-frames.md: `plan` with a fake `fetch` against a transcription of (3),
read off its plan shards."""

import collections

import pytest
from vpt_checks import plan_config, read_shards, totals
from vpt_world import (
    BAD_JSONL,
    Ep,
    eligible,
    joined,
    lines,
    make_world,
    options,
    reference_draws,
    reference_shards,
)

from mcpref.vpt.gui_frames import plan

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


def read_episodes(directory) -> list[dict]:
    return joined(read_shards(directory), "episodes")


def assert_plan(world, cfg, directory) -> None:
    """`plan` writes the shards of the transcription of (3) and returns their totals."""
    returned = plan(cfg, world.fetch, directory)
    draws = reference_draws(world, cfg.n_frames, cfg.per_episode, cfg.seed)
    expected = reference_shards(draws, cfg.plan_shard_draws)
    assert read_shards(directory) == expected
    assert returned == totals(expected)


@pytest.mark.parametrize("n", [1, 4, 9, 100])
@pytest.mark.parametrize("m", [1, 3])
def test_plan_matches_transcription_eq3(tmp_path, n, m):
    """Subset choice, draws within a subset until valid, pool exhaustion, the cut at N."""
    world = make_world(MIXED)
    for seed in SEEDS:
        cfg = plan_config(world, n_frames=n, per_episode=m, seed=seed)
        assert_plan(world, cfg, tmp_path / str(seed))


def test_plan_draws_subsets_in_config_order(tmp_path):
    world = make_world(MIXED, order=("10xx", "8xx", "7xx", "6xx", "9xx"))
    for seed in SEEDS:
        cfg = plan_config(world, n_frames=7, per_episode=2, seed=seed)
        assert_plan(world, cfg, tmp_path / str(seed))


def test_plan_exhausts_every_pool_once_without_repeats(tmp_path):
    world = make_world(MIXED)
    valid = [e for es in MIXED.values() for e in set(es) if e.reason is None]
    for seed in SEEDS:
        cfg = plan_config(world, n_frames=1000, per_episode=2, seed=seed)
        n, _ = plan(cfg, world.fetch, tmp_path / str(seed))
        drawn = [
            (r["subset"], r["relpath"]) for r in read_episodes(tmp_path / str(seed))
        ]
        assert sorted(drawn) == sorted(world.eps)
        assert n == sum(min(2, len(eligible(e.gui))) for e in valid)


@pytest.mark.parametrize("n", range(1, 14))
def test_plan_stops_at_n_frames(tmp_path, n):
    """len = min(N, all available); the last drawn episode is the one completing N."""
    world = make_world(MIXED)
    for seed in SEEDS:
        cfg = plan_config(world, n_frames=n, per_episode=3, seed=seed)
        planned, _ = plan(cfg, world.fetch, tmp_path / str(seed))
        frames = joined(read_shards(tmp_path / str(seed)), "frames")
        assert planned == len(frames) == min(n, 13)
        assert n > 13 or read_episodes(tmp_path / str(seed))[-1]["reason"] is None


def test_plan_frames_distinct_eligible_and_paired_with_lines(tmp_path):
    """Frame i of an episode: i in F_e, prev_line = l_{i-1}, line = l_i, id of Interface."""
    world = make_world(MIXED)
    for seed in SEEDS:
        cfg = plan_config(world, n_frames=100, per_episode=3, seed=seed)
        plan(cfg, world.fetch, tmp_path / str(seed))
        by_episode = collections.defaultdict(list)
        for p in joined(read_shards(tmp_path / str(seed)), "frames"):
            e = world.eps[p["subset"], p["relpath"]]
            i = p["frame"]
            assert p["id"] == f"{e.version}_{e.stem}_{i}" and p["version"] == e.version
            assert (p["prev_line"], p["line"]) == (lines(e.gui)[i - 1], lines(e.gui)[i])
            by_episode[p["subset"], p["relpath"]].append(i)
        for key, got in by_episode.items():
            f_e = eligible(world.eps[key].gui)
            assert len(set(got)) == len(got) == min(3, len(f_e))
            assert set(got) <= set(f_e)
    assert not [u for u in world.fetched if u.endswith(".mp4")]  # (2): .mp4 not checked
