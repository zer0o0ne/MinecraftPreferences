"""(6), (7), (9) of specs/perception-data.md: runs interrupted before, inside and after the
write of a shard (the classifier, a fetch or a storage call raising; a kill) keep the
complete shards; the continuation draws on from the last one without redoing its work and
ends with the shards of an uninterrupted run."""

import json
import os
import shutil

import pytest

from .collect_gamma import Classifier
from .collect_out import assert_outputs
from .collect_site import asked_files

K = 2  # seed 0: draw 2 pops old6 (version), then a6 (two clips); draws 0, 1 have candidates


def names(k: int) -> list[str]:
    return [f"{j:06d}" for j in range(k)]


def continued(site, h: int) -> Classifier:
    """Continues the interrupted run: it ends as the uninterrupted one, without a request,
    decode or classification for an episode of shards 0..h-1; returns the classifier."""
    ex = site.expected()
    asked, decoded = len(site.server.requests), len(site.decodes())
    gamma = site.run()
    assert_outputs(site, ex)
    done = [r for s in ex.shards[:h] for r in s.records]
    assert asked_files(site, [r["relpath"] for r in done], asked) == []
    stems = {r["relpath"].split("/")[2][:-4] for r in done}
    assert not stems & (
        {d["key"] for d in site.decodes()[decoded:]} | set(gamma.by_stem())
    )
    return gamma


@pytest.mark.parametrize("workers", [1, 3])
def test_classifier_raising_in_a_draw(site, workers):
    """A worker's exception is raised once shards 0..k-1 are written; the continuation
    redoes draw k."""
    gamma = site.aborted(site.failing_gamma(K), workers=workers)
    assert site.error is gamma.raised and not site.completed()
    assert site.complete() == names(K)
    assert "a6" in continued(site, K).by_stem()


@pytest.mark.parametrize("workers", [1, 3])
def test_video_fetch_failing_in_a_draw(site, workers):
    """HTTP 503 on the HEAD of a6's video after the retries, in its worker."""
    path = site.path("data/6.12/a6.mp4", ".mp4")
    site.faults[site.head(path)] = [503] * 2
    site.aborted(workers=workers)
    assert site.asked(path) == [site.head(path)] * 2 and site.complete() == names(K)
    site.faults.clear()
    assert "a6" in continued(site, K).by_stem()


@pytest.mark.parametrize("workers", [1, 3])
def test_planner_fetch_failing_in_a_draw(site, workers):
    """HTTP 503 on the options file of b7 (draw 3) after the retries: raised by the planner
    at once; with one worker shards 0..2 are written before draw 3 starts."""
    path = site.path("data/7.1/b7.mp4", "-options.json")
    site.faults[site.head(path)] = [503] * 2
    site.aborted(workers=workers)
    h = len(site.complete())
    assert site.complete() == names(h) and h <= 3 and (h == 3 or workers > 1)
    site.faults.clear()
    continued(site, h)


def fail_fsync(monkeypatch, when) -> list:
    """os.fsync raises OSError at its first call on a path for which `when` holds; returns
    the exceptions raised."""
    real, raised = os.fsync, []

    def fsync(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if not raised and when(path):
            raised.append(OSError(28, "No space left on device", path))
            raise raised[-1]
        return real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return raised


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize(
    "file", ["records.json", "state.json", "options", "frames.yuv"]
)
def test_storage_failing_inside_a_shard_write(site, monkeypatch, workers, file):
    """The sync of a file of staged shard k raises: shards 0..k-1 stay, the staged shard
    is left; the continuation writes shard k again."""
    staged = str((site.out / "shards" / f"{K:06d}.part").resolve())
    raised = fail_fsync(monkeypatch, lambda p: p.startswith(staged + "/") and file in p)
    site.aborted(workers=workers)
    assert site.error is raised[0]
    assert site.shard_names() == names(K) + [f"{K:06d}.part"]
    assert "a6" in continued(site, K).by_stem()


@pytest.mark.parametrize("workers", [1, 3])
def test_storage_failing_after_a_shard_write(site, monkeypatch, workers):
    """The sync of shards/ after the rename of shard k raises: shard k is complete; the
    continuation starts with draw k + 1."""
    shards = site.out.resolve() / "shards"
    raised = fail_fsync(monkeypatch, lambda p: p == str(shards) and (shards / f"{K:06d}").is_dir())  # fmt: skip
    site.aborted(workers=workers)
    assert site.error is raised[0] and site.complete() == names(K + 1)
    gamma = continued(site, K + 1)
    assert "a6" not in gamma.by_stem()


def killed(site, h: int) -> None:
    """Turns the outputs of a complete run into what a kill inside the write of shard h
    leaves: shards 0..h-1, a staged shard h, staging with a video and a clip of draw h,
    run.json without counts."""
    root = site.out / "shards"
    for name in site.shard_names()[h:]:
        shutil.move(root / name, root / f"{name}.gone")
    shutil.move(root / f"{h:06d}.gone", root / f"{h:06d}.part")
    for path in root.glob("*.gone"):
        shutil.rmtree(path)
    (root / f"{h:06d}.part" / "records.json").write_text("[")
    (site.out / "staging" / str(h) / "6.12_a6_5.part").mkdir(parents=True)
    (site.out / "staging" / str(h) / "6.12_a6_5.part" / "frames.yuv").write_bytes(b"x")
    (site.out / "staging" / f"{h}.mp4").write_bytes(b"FAKEMP4 a6\n")
    run = site.run_json()
    del run["episodes"], run["clips"]
    (site.out / "run.json").write_text(json.dumps(run))


@pytest.mark.parametrize("h", [0, 1, K, 10])
def test_continuation_after_a_kill(site, h):
    site.run()
    killed(site, h)
    assert not site.completed()
    continued(site, h)


def test_continuation_of_a_complete_run_does_nothing(site):
    site.run()
    before, asked = site.raw(), len(site.server.requests)
    gamma = site.run()
    assert site.raw() == before and gamma.calls == [] and len(site.server.requests) == asked  # fmt: skip


def test_continuation_after_two_interruptions(site):
    """Aborted in draw 1, then in draw 4, then continued."""
    site.aborted(site.failing_gamma(1))
    assert site.complete() == names(1)
    site.aborted(site.failing_gamma(4), workers=3)
    assert site.complete() == names(4)
    continued(site, 4)
