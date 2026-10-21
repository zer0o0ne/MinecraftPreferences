"""(W), (P), (9') of specs/preference-pool.md: runs interrupted inside a draw (a worker's
fetch failing), inside and after the write of a shard (a storage call raising; a kill)
keep the complete shards; the continuation draws on from the last one without asking the
heads again or redoing the work of the complete shards, and ends with the outputs of an
uninterrupted run."""

import json
import shutil

import pytest
from clips.test_collect_interrupt import fail_fsync

from .pool_site import assert_outputs
from .pool_world import small_episodes

COUNTS = ("episodes", "sessions", "clips", "comparisons")


@pytest.fixture
def site(make_site):
    """The small world: five draws."""
    return make_site(episodes=small_episodes())


def first_valid(ex, after: int) -> int:
    """The first draw from `after` on whose first episode is valid (has a video)."""
    return next(k for k, s in enumerate(ex.shards) if k >= after and s.records[0]["reason"] is None)  # fmt: skip


def names(k: int) -> list[str]:
    return [f"{j:06d}" for j in range(k)]


def continued(site, h: int) -> None:
    """The continuation ends as the uninterrupted run, with no head request and no
    request or decode for an episode of shards 0..h-1."""
    ex = site.expected()
    asked, decoded = len(site.server.requests), len(site.decodes())
    site.run()
    assert_outputs(site, ex)
    assert site.heads_asked(asked) == []
    done = [r["relpath"] for s in ex.shards[:h] for r in s.records]
    assert site.worked(done, asked) == set()
    stems = {r.split("/")[2][:-4] for r in done}
    assert not stems & {d["key"] for d in site.decodes()[decoded:]}


@pytest.mark.parametrize("workers", [1, 3])
def test_video_fetch_failing_in_a_draw(site, workers):
    """HTTP 503 on the HEAD of draw k's first video after the retries, in its worker:
    raised once shards 0..k-1 are written; the continuation redoes draw k."""
    k = first_valid(site.expected(), 2)
    path = site.path(site.expected().shards[k].records[0]["relpath"], ".mp4")
    site.faults[site.head(path)] = [503] * 2
    site.aborted(workers=workers)
    assert site.asked(path) == [site.head(path)] * 2
    assert site.complete() == names(k) and not site.completed()
    site.faults.clear()
    continued(site, k)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("file", ["records.json", "state.json", "comparisons.json", "options", "frames.yuv"])  # fmt: skip
def test_storage_failing_inside_a_shard_write(site, monkeypatch, workers, file):
    """The sync of a file of staged shard k (one with clips) raises: shards 0..k-1 stay,
    the staged shard is left; the continuation writes shard k again."""
    k = next(j for j, s in enumerate(site.expected().shards) if j and s.comparisons)
    staged = str((site.out / "shards" / f"{k:06d}.part").resolve())
    raised = fail_fsync(monkeypatch, lambda p: p.startswith(staged + "/") and file in p)
    site.aborted(workers=workers)
    assert site.error is raised[0]
    assert site.shard_names() == names(k) + [f"{k:06d}.part"]
    monkeypatch.undo()
    continued(site, k)


@pytest.mark.parametrize("workers", [1, 3])
def test_storage_failing_after_a_shard_write(site, monkeypatch, workers):
    """The sync of shards/ after the rename of shard 2 raises: shard 2 is complete; the
    continuation starts with draw 3."""
    shards = site.out.resolve() / "shards"
    raised = fail_fsync(monkeypatch, lambda p: p == str(shards) and (shards / "000002").is_dir())  # fmt: skip
    site.aborted(workers=workers)
    assert site.error is raised[0] and site.complete() == names(3)
    monkeypatch.undo()
    continued(site, 3)


def killed(site, h: int) -> None:
    """Turns the outputs of a complete run into what a kill inside the write of shard h
    leaves: shards 0..h-1, a staged shard h, staging with a video and a clip of draw h,
    run.json without counts."""
    root = site.out / "shards"
    for name in site.shard_names()[h + 1 :]:
        shutil.rmtree(root / name)
    shutil.move(root / f"{h:06d}", root / f"{h:06d}.part")
    (root / f"{h:06d}.part" / "records.json").write_text("[")
    (site.out / "staging" / str(h) / "x_1.part").mkdir(parents=True)
    (site.out / "staging" / str(h) / "x_1.part" / "frames.yuv").write_bytes(b"x")
    (site.out / "staging" / str(h) / "0.mp4").write_bytes(b"FAKEMP4 x\n")
    run = site.run_json()
    for key in COUNTS:
        del run[key]
    (site.out / "run.json").write_text(json.dumps(run))


@pytest.mark.parametrize("h", [0, 1, 4])
def test_continuation_after_a_kill(site, h):
    site.run()
    killed(site, h)
    assert not site.completed()
    continued(site, h)


def test_continuation_of_a_complete_run_does_nothing(site):
    site.run()
    before, asked, decoded = site.raw(), len(site.server.requests), len(site.decodes())
    site.run()
    assert site.raw() == before and len(site.server.requests) == asked
    assert len(site.decodes()) == decoded


def test_continuation_after_two_interruptions(site):
    """Aborted in the head pass (a head answering 403), then in a draw, then continued."""
    ex = site.expected()
    first = ex.entries[2][0]
    head = ("GET", site.path(first, ".jsonl"), "bytes=0-1000")
    site.faults[head] = [403]
    site.aborted()
    assert sorted(site.raw("sessions")) == [f"sessions/{k:06d}/sessions.json" for k in range(2)]  # fmt: skip
    k = first_valid(ex, 2)
    path = site.path(ex.shards[k].records[0]["relpath"], ".mp4")
    site.faults[site.head(path)] = [503] * 2
    site.aborted(workers=3)
    assert site.complete() == names(k)
    site.faults.clear()
    continued(site, k)
