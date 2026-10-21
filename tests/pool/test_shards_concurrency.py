"""(W) and (9') of specs/preference-pool.md under concurrency: the planner starts draw k'
only while fewer than `workers` episodes of earlier draws are unwritten; after the shard
reaching n_clips the running work is ignored; a worker's exception is raised in draw
order, the first failing episode of its draw in episode order, the earlier shards
written."""

import threading
import time

import pytest
from clips.collect_gamma import Classifier

from .pool_site import assert_outputs
from .pool_world import chain, relpath


def wait(condition, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while not condition() and time.monotonic() < end:
        time.sleep(0.02)
    return condition()


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_lookahead_bound(site, workers):
    """The first episode of draw a (the first draw from 1 on starting with a valid
    episode) holds its video: shards 0..a-1 are written; the draws started are a..b, b the
    last with fewer than `workers` episodes of draws a..b-1; with a free thread all their
    episodes run, with one worker only the held one; no later draw is worked on."""
    ex = site.expected()
    a = next(k for k, s in enumerate(ex.shards) if k and s.records[0]["reason"] is None)
    sizes = [len(s.records) for s in ex.shards]
    b = max(k for k in range(a, len(sizes)) if sum(sizes[a:k]) < workers)
    held = ex.shards[a].records[0]["relpath"]
    started = [r["relpath"] for s in ex.shards[a : b + 1] for r in s.records]
    later = [r["relpath"] for s in ex.shards[b + 1 :] for r in s.records]
    gate = threading.Event()
    site.gates[site.path(held, ".mp4")] = gate
    thread = site.in_thread(Classifier(site.world), workers=workers)
    try:
        want = set(started) if workers > 1 else {held}
        assert wait(lambda: site.worked(started) == want), site.worked(started)
        assert wait(lambda: site.complete() == [f"{k:06d}" for k in range(a)])
        time.sleep(0.3)
        assert later and site.worked(later) == set()
        assert site.worked(started) == want
        assert site.complete() == [f"{k:06d}" for k in range(a)]
    finally:
        gate.set()
        thread.join(10.0)
    assert site.error is None
    assert_outputs(site, ex)


def cut_world() -> dict:
    """One subset: session A (two `w`*16 pieces, four clips), session B (its first piece
    opens the inventory: the classifier is called)."""
    a = chain("9.1", "a", "ab" * 6, ["w" * 16] * 2)
    b = chain("9.1", "b", "ab" * 6, ["wwiiwwwwwwww", "w" * 6])
    return {"9xx": a + b}


def test_stop_ignores_the_running_work(make_site):
    """n_clips 3, three workers, a seed drawing A first: A's video is held until the
    classifier has raised for B, started meanwhile; the run ends normally with shard 0,
    every decode ended, staging removed."""
    site = make_site(episodes=cut_world())
    seed = next(s for s in range(50) if site.expected(seed=s).shards[0].records[0]["relpath"].split("/")[2].startswith("a-"))  # fmt: skip
    ex = site.expected(seed=seed, n_clips=3)
    assert len(ex.shards) == 1 and ex.shards[0].clips == 4
    gamma = Classifier(site.world, fail_stem=relpath("9.1", "b", "ab" * 6, 0).split("/")[2][:-4])  # fmt: skip
    site.gates[site.path(ex.shards[0].records[0]["relpath"], ".mp4")] = gamma.raised_now
    site.run(gamma, seed=seed, n_clips=3, workers=3)
    assert gamma.raised_now.is_set()
    assert_outputs(site, ex)
    assert len(site.decodes()) == len(site.decodes("end"))


def video_world() -> dict:
    """Session x (one `w`*6 piece) in 6xx, session y (two `w`*6 pieces) in 8xx."""
    return {"6xx": chain("6.13", "x", "ab" * 6, ["w" * 6]),
            "8xx": chain("8.1", "y", "ab" * 6, ["w" * 6] * 2)}  # fmt: skip


@pytest.mark.parametrize("workers", [1, 3])
def test_first_failing_episode_of_the_draw_is_raised(make_site, workers):
    """Both videos of session y answer 403 to their HEAD, the first one later: the
    exception names the first episode's video; the draws before y are written."""
    site = make_site(episodes=video_world())
    seed = next(s for s in range(50) if site.expected(seed=s).shards[0].records[0]["relpath"].split("/")[2].startswith("x-"))  # fmt: skip
    y0, y1 = (relpath("8.1", "y", "ab" * 6, k) for k in range(2))
    for rp in (y0, y1):
        site.faults[site.head(site.path(rp, ".mp4"))] = [403] * 3
    site.stalls[site.head(site.path(y0, ".mp4"))] = 0.3
    site.aborted(seed=seed, workers=workers)
    assert site.path(y0, ".mp4") in str(site.error)
    assert site.path(y1, ".mp4") not in str(site.error)
    assert site.complete() == ["000000"] and not site.completed()
