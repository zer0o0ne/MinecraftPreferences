"""(2), (5), (9) of specs/perception-data.md: the video's ranged fetch into
staging/<k>.mp4, the two decodes (command, success = exit 0 with exactly T + 1 frames,
retries, reason `video`), a timed-out decode killed and aborting the run with TimeoutError,
the failure of pass 2, ffmpeg's stderr; with the fake ffmpeg of fake_decoder.py."""

import itertools
import os
import time

import pytest
from progress_bars import Bar, screen
from vpt_checks import part_ranges

from .collect_out import assert_outputs
from .collect_world import EPISODES, Ep

SCRIPT = "w" * 8  # starts 0..3
INDICES, RESUME = Bar("indices", 5, 5, "file"), "resume: nothing to do"


def one(make_site, **kw):
    return make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", SCRIPT, **kw)]})


def command(site, k: int) -> list[str]:
    video = os.path.realpath(site.out / "staging" / f"{k}.mp4")
    return [site.fields["ffmpeg"], "-nostdin", "-loglevel", "error", "-i", video,
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]  # fmt: skip


def test_each_episode_with_clips_is_decoded_twice_with_the_command(site):
    """The video of the candidate of draw k is decoded from {out_dir}/staging/<k>.mp4."""
    site.run(workers=3)
    ex = site.expected()
    runs = {}
    for d in site.decodes():
        argv = list(d["argv"])
        argv[5] = os.path.realpath(argv[5])
        runs.setdefault(d["key"], []).append(argv)
    stems = {e.stem: e.relpath for e in site.world.eps.values()}
    with_clips = {r["relpath"] for r in ex.records if r["clips"]}
    assert {stems[x] for x in runs} == {r["relpath"] for r in ex.records if r["reason"] in (None, "no_clip")}  # fmt: skip
    for stem, argvs in runs.items():
        assert all(a == command(site, ex.draw[stems[stem]]) for a in argvs), stem
        assert len(argvs) == 1 + (stems[stem] in with_clips), stem


def test_video_is_fetched_by_ranges_after_one_head(site):
    site.run()
    path = site.path("data/6.12/a6.mp4", ".mp4")
    size = len(site.data(path))
    parts = [site.part(path, j) for j in range(len(part_ranges(size, 3)))]
    asked = site.asked(path)
    assert asked[0] == site.head(path) and sorted(asked[1:]) == sorted(parts)
    heads = [r.headers for r in site.server.requests if r.path == path]
    assert all(h["x-ms-version"] == "2019-12-12" for h in heads)


@pytest.mark.parametrize("session", [False, True])
def test_missing_video_is_reason_video(make_site, session):
    """starts_session is computed from the lines before the video is fetched."""
    site = make_site(episodes={"9xx": [Ep("data/9.1/v9.mp4", video=None, session=session)]})  # fmt: skip
    site.run()
    (rec,) = site.records()
    assert rec["reason"] == "video" and rec["options"] is not None
    assert rec["starts_session"] is session and rec["valid_starts"] is None
    path = site.path("data/9.1/v9.mp4", ".mp4")
    assert site.asked(path) == [site.head(path)] and site.decodes() == []
    assert_outputs(site, site.expected())


@pytest.mark.parametrize("failure", ["exit1", "short", "long", "partial"])
@pytest.mark.parametrize("retries", [0, 2])
def test_decode_failing_every_attempt_is_reason_video(make_site, failure, retries):
    site = one(make_site)
    site.behaviours = {"a": [failure] * (retries + 1)}
    gamma = site.run(http_retries=retries)
    (rec,) = site.records()
    assert rec == site.expected(broken={"a"}).records[0] and rec["reason"] == "video"
    assert len(site.decodes()) == retries + 1 and gamma.calls == []
    assert_outputs(site, site.expected(broken={"a"}))


@pytest.mark.parametrize("failure", ["exit1", "short", "long", "partial"])
def test_decode_retried_after_retry_delay(make_site, failure):
    site = one(make_site)
    site.behaviours = {"a": [failure, failure, "ok"]}
    site.run(http_retries=2, retry_delay=0.3)
    assert_outputs(site, site.expected())
    starts, ends = site.decodes(), site.decodes("end")
    assert len(starts) == 4  # pass 1 three times, pass 2 once
    assert all(b["t"] - a["t"] >= 0.3 for a, b in zip(ends[:2], starts[1:3]))


def gone(pid: int) -> bool:
    """Whether the process `pid` has ended (within 2 s)."""
    end = time.monotonic() + 2.0
    while os.path.exists(f"/proc/{pid}") and time.monotonic() < end:
        time.sleep(0.02)
    return not os.path.exists(f"/proc/{pid}")


@pytest.mark.parametrize("plan", [["slow"], ["ok", "slow"]], ids=["pass_1", "pass_2"])
def test_decode_over_ffmpeg_timeout_is_killed_and_aborts(make_site, plan):
    """Not retried, no reason given; the video deleted; a rerun tries again."""
    site = one(make_site)
    site.behaviours = {"a": plan}
    site.aborted(ffmpeg_timeout=1.5)
    assert isinstance(site.error, TimeoutError)
    assert (
        len(site.decodes()) == len(plan) and len(site.decodes("end")) == len(plan) - 1
    )
    assert gone(site.decodes()[-1]["pid"])
    assert site.complete() == [] and not (site.out / "staging" / "0.mp4").exists()
    assert not site.completed()
    site.run(ffmpeg_timeout=1.5)
    assert_outputs(site, site.expected())


def test_ffmpeg_not_starting_aborts_the_run(site):
    """No program at `ffmpeg`: draw 0's candidate (r10) aborts the run, no shard written;
    a rerun with the program (not an output field) continues."""
    missing = site.tmp / "no" / "ffmpeg"
    site.aborted(ffmpeg=str(missing))
    assert isinstance(site.error, OSError) and site.complete() == []
    site.run()
    assert_outputs(site, site.expected())


def test_failing_pass_2_aborts_the_run(make_site):
    """Pass 2 exits with 1: RuntimeError, not retried; a rerun decodes both passes again."""
    site = one(make_site)
    site.behaviours = {"a": ["ok", "exit1"]}
    site.aborted(http_retries=2)
    assert type(site.error) is RuntimeError
    assert len(site.decodes()) == 2 and site.complete() == [] and not site.completed()
    site.run()
    assert len(site.decodes()) == 4
    assert_outputs(site, site.expected())


@pytest.mark.parametrize("failure", ["short", "long", "partial"])
def test_pass_2_with_a_wrong_byte_count_aborts(make_site, failure):
    site = one(make_site)
    site.behaviours = {"a": ["ok", failure]}
    site.aborted()
    assert type(site.error) is RuntimeError and site.complete() == []


def test_ffmpeg_stderr_is_printed_above_the_bar(make_site, capfd):
    """One tqdm.write per run with stderr: decoded with replacement, trailing newlines
    removed; nothing for a run with empty stderr."""
    site = one(make_site)
    site.behaviours = {"a": ["exit1", "ok", "ok"]}
    site.noise = {"a": [b"first try\n", b"bad \xff byte\n\n", b""]}
    site.run()
    shown = screen(capfd.readouterr().err)
    assert shown[:4] == [
        INDICES,
        RESUME,
        "first try",
        "bad \N{REPLACEMENT CHARACTER} byte",
    ]
    assert len(shown) == 5 and shown[4].desc == "collect"


def test_timed_out_decode_prints_its_stderr(make_site, capfd):
    site = one(make_site)
    site.behaviours, site.noise = {"a": ["slow"]}, {"a": [b"stuck \xff\n"]}
    site.aborted(ffmpeg_timeout=1.0)
    shown = screen(capfd.readouterr().err)
    assert shown[:3] == [INDICES, RESUME, "stuck \N{REPLACEMENT CHARACTER}"]
    assert [x.desc for x in shown[3:]] == ["collect"]


@pytest.mark.parametrize("workers", [1, 3])
def test_episodes_are_processed_on_up_to_workers_threads(make_site, workers):
    """Four candidates, each decode held 0.15 s: at most `workers` decodes at once, more
    than one with three workers."""
    eps = {s: [Ep(f"data/7.0/{x}.mp4", "w" * 6)] for s, x in zip(EPISODES, "abcd")}
    site = make_site(episodes=eps)
    site.behaviours, site.sleep = {s: ["slow", "slow"] for s in "abcd"}, 0.15
    site.run(workers=workers)
    assert_outputs(site, site.expected())
    events = sorted([(d["t"], 1) for d in site.decodes()]
                    + [(d["t"], -1) for d in site.decodes("end")])  # fmt: skip
    running = max(itertools.accumulate(e for _, e in events))
    assert running == 1 if workers == 1 else 2 <= running <= workers
