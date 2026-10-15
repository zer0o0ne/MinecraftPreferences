"""(3') of specs/vpt-gui-frames.md: plan shards of K draws, interruption of `plan` (a `fetch`
raising) and its continuation with any K, against the transcription of (3)."""

import collections
import dataclasses
from pathlib import Path

import pytest
from vpt_checks import plan_config, read_shards, totals
from vpt_world import (
    FIELDS,
    SUBSETS,
    Ep,
    make_world,
    options,
    reference_draws,
    reference_shards,
)

from mcpref.vpt.gui_frames import plan

KINDS = (None, "missing", None, "version", "no_gui", None, "empty", "options")
FILES = {"missing": {"options": None}, "empty": {"jsonl": b""}, "no_gui": {"gui": "#.#.#"},
         "options": {"options": options(lang="en_gb")}}  # fmt: skip
WHERE = ["start", "boundary", "shard", "pop", "draw"]
ENDS = {"N": 12, "pools": 100}  # n_frames ending (3) by N or by empty pools (m = 2)
SIZES = {"1": lambda d: 1, "D": lambda d: d, "D + 1": lambda d: d + 1, "4": lambda d: 4}


def ep(name: str, reason: str | None) -> Ep:
    version = "6.9" if reason == "version" else "7.1"
    fields = {"gui": "##.###"} | FILES.get(reason, {})  # F_e = [1, 4, 5]
    return Ep(f"data/{version}/{name}.mp4", reason=reason, **fields)


LONG = {s: [ep(f"{s}-{k}", KINDS[(3 * j + k) % 8]) for k in range(7)]
        for j, s in enumerate(SUBSETS)}  # fmt: skip
LONG["7xx"].append(LONG["7xx"][0])  # an exact repeat


class Interrupted(Exception):
    pass


def point(draws: list[dict], where: str, k: int, start: int = 0) -> tuple | None:
    """(d, record, nth): a run of K = k from draw `start` stopped in draw d at the nth fetch of
    a file of `record`'s episode. `start`: in the run's first draw; `boundary`: the first draw
    of a shard after a complete one; `shard`: a later draw of a shard after a complete one;
    `pop`: as `shard`, at its second downloaded episode; `draw`: as `shard`, at the second
    file of its valid episode. None if `draws` have no such point."""
    for d in range(start, len(draws)):
        draw, j = draws[d], d - start
        fetched = [r for r in draw["episodes"] if r["reason"] != "version"]
        inside = j > k and j % k
        if fetched and (where == "start" and j == 0 or where == "shard" and inside
                        or where == "boundary" and j and not j % k):  # fmt: skip
            return d, fetched[0], 1
        if where == "pop" and inside and len(fetched) >= 2:
            return d, fetched[1], 1
        if where == "draw" and inside and draw["frames"]:
            return d, draw["episodes"][-1], 2
    return None


def find(world, n: int, where: str, k: int) -> tuple[int, list, tuple]:
    """The first seed whose draws of (3) have the point `where`: seed, draws, point."""
    for seed in range(200):
        draws = reference_draws(world, n, FIELDS["per_episode"], seed)
        if found := point(draws, where, k):
            return seed, draws, found
    raise AssertionError(f"no seed with a point {where}")


def raising(world, record: dict, nth: int):
    """`world.fetch`, raising Interrupted at the nth fetch of a file of `record`."""
    e = world.eps[record["subset"], record["relpath"]]
    urls = {world.url(record["subset"], e, s) for s in ("-options.json", ".jsonl")}
    seen = []

    def fetch(url: str) -> bytes | None:
        seen.extend([url] if url in urls else [])
        if len(seen) == nth:
            raise Interrupted
        return world.fetch(url)

    return fetch


def stop(world, cfg, found: tuple, directory: Path) -> None:
    with pytest.raises(Interrupted):
        plan(cfg, raising(world, *found[1:]), directory)


@pytest.mark.parametrize("end", ENDS)
@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("where", WHERE)
def test_continued_plan_equals_uninterrupted(tmp_path, where, k, end):
    """K = 3 stopped at `where` keeps its complete shards; continued with K = k: every
    shard as (3) run without interruption cut at those points, with its generator state."""
    world = make_world(LONG)
    seed, draws, found = find(world, ENDS[end], where, 3)
    cfg = plan_config(world, n_frames=ENDS[end], seed=seed, plan_shard_draws=3)
    stop(world, cfg, found, tmp_path)
    h = found[0] // 3
    assert read_shards(tmp_path) == reference_shards(draws, 3)[:h]
    returned = plan(dataclasses.replace(cfg, plan_shard_draws=k), world.fetch, tmp_path)
    expected = reference_shards(draws, k, ((h, 3),))
    assert read_shards(tmp_path) == expected
    assert returned == totals(expected)


def test_plan_interrupted_twice_then_completed(tmp_path):
    """K = 3 stopped inside a shard, K = 2 stopped inside a draw, K = 5 to the end."""
    world = make_world(LONG)
    seed, draws, first = find(world, ENDS["pools"], "shard", 3)
    h1 = first[0] // 3
    second = point(draws, "draw", 2, start=3 * h1)
    h2 = (second[0] - 3 * h1) // 2
    cfg = plan_config(world, n_frames=ENDS["pools"], seed=seed, plan_shard_draws=3)
    stop(world, cfg, first, tmp_path)
    stop(world, dataclasses.replace(cfg, plan_shard_draws=2), second, tmp_path)
    assert read_shards(tmp_path) == reference_shards(draws, 2, ((h1, 3),))[: h1 + h2]
    returned = plan(dataclasses.replace(cfg, plan_shard_draws=5), world.fetch, tmp_path)
    expected = reference_shards(draws, 5, ((h1, 3), (h2, 2)))
    assert read_shards(tmp_path) == expected and returned == totals(expected)


def snapshot(directory: Path) -> dict:
    return {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("end", ENDS)
@pytest.mark.parametrize("k", SIZES)
def test_finished_plan_continues_to_nothing(tmp_path, end, k):
    """K = 1 and K = D (the number of draws, 6 or 15) end on a shard boundary, no shard is
    written without a draw; K = D + 1 and K = 4 end inside a shard. A further `plan` fetches
    nothing but the indices, changes nothing and returns the same totals."""
    world = make_world(LONG)
    draws = reference_draws(world, ENDS[end], FIELDS["per_episode"], 0)
    size = SIZES[k](len(draws))
    cfg = plan_config(world, n_frames=ENDS[end], plan_shard_draws=size)
    returned = plan(cfg, world.fetch, tmp_path)
    assert read_shards(tmp_path) == reference_shards(draws, size)
    before, world.fetched = snapshot(tmp_path), []
    assert plan(cfg, world.fetch, tmp_path) == returned
    assert snapshot(tmp_path) == before
    assert set(world.fetched) <= set(world.index_urls.values())


def test_every_pool_empty_writes_no_shard(tmp_path):
    world = make_world({})
    cfg = plan_config(world)
    for _ in range(2):
        assert plan(cfg, world.fetch, tmp_path / "plan") == (0, collections.Counter())
        assert read_shards(tmp_path / "plan") == []
