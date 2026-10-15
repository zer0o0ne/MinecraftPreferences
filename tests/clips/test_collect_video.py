"""(2), (5), (8) of specs/perception-data.md: the video's ranged fetch into staging, the
two decodes (command, success = exit 0 in time with exactly T + 1 frames, retries, reason
`video`), the failure of pass 2, ffmpeg's stderr; with the fake ffmpeg of fake_decoder.py."""

import itertools
import os

import pytest
from progress_bars import Bar, screen
from vpt_checks import part_ranges

from .collect_world import Ep

SCRIPT = "w" * 8  # starts 0..3


def one(make_site, **kw):
    return make_site(episodes={"6xx": [Ep("data/7.0/a.mp4", SCRIPT, **kw)]})


def command(site, r: int) -> list[str]:
    video = os.path.realpath(site.out / "staging" / f"{r}.mp4")
    return [site.fields["ffmpeg"], "-nostdin", "-loglevel", "error", "-i", video,
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]  # fmt: skip


def test_each_episode_with_clips_is_decoded_twice_with_the_command(site):
    """The video of draw position r is decoded from {out_dir}/staging/<r>.mp4."""
    site.run()
    ex = site.expected()
    runs = {}
    for d in site.decodes():
        argv = list(d["argv"])
        argv[5] = os.path.realpath(argv[5])
        runs.setdefault(d["key"], []).append(argv)
    with_clips = {r["relpath"].split("/")[2][:-4] for r in ex.records if r["clips"]}
    assert with_clips <= set(runs) and len(with_clips) == 4
    for stem, argvs in runs.items():
        assert all(a == command(site, ex.r[stem]) for a in argvs), stem
        assert len(argvs) == 2 or stem not in with_clips


def test_video_is_fetched_by_ranges_after_one_head(site):
    site.run()
    path = site.path("6xx", "data/6.12/a6.mp4", ".mp4")
    size = len(site.data(path))
    parts = [site.part(path, j) for j in range(len(part_ranges(size, 3)))]
    asked = site.asked(path)
    assert asked[0] == site.head(path) and sorted(asked[1:]) == sorted(parts)
    heads = [r.headers for r in site.server.requests if r.path == path]
    assert all(h["x-ms-version"] == "2019-12-12" for h in heads)


def test_missing_video_is_reason_video(site):
    site.run()
    (rec,) = [
        r for r in site.read_jsonl("episodes.jsonl") if r["relpath"].endswith("v9.mp4")
    ]
    assert rec["reason"] == "video" and rec["options"] is not None
    assert rec["starts_session"] is False and rec["valid_starts"] is None
    path = site.path("9xx", "data/9.1/v9.mp4", ".mp4")
    assert site.asked(path) == [site.head(path)]
    assert "v9" not in {d["key"] for d in site.decodes()}


@pytest.mark.parametrize("failure", ["exit1", "short", "long", "partial"])
@pytest.mark.parametrize("retries", [0, 2])
def test_decode_failing_every_attempt_is_reason_video(make_site, failure, retries):
    site = one(make_site)
    site.behaviours = {"a": [failure] * (retries + 1)}
    gamma = site.run(http_retries=retries)
    (rec,) = site.read_jsonl("episodes.jsonl")
    assert rec == site.expected(broken={"a"}).records[0] and rec["reason"] == "video"
    assert len(site.decodes()) == retries + 1 and gamma.calls == []
    assert site.tree("staging") == {} and site.tree("clips") == {}


@pytest.mark.parametrize("failure", ["exit1", "short", "long", "partial"])
def test_decode_retried_after_retry_delay(make_site, failure):
    site = one(make_site)
    site.behaviours = {"a": [failure, failure, "ok"]}
    site.run(http_retries=2, retry_delay=0.3)
    assert site.read_jsonl("clips.jsonl") == site.expected().clips
    starts, ends = site.decodes(), site.decodes("end")
    assert len(starts) == 4  # pass 1 three times, pass 2 once
    assert all(b["t"] - a["t"] >= 0.3 for a, b in zip(ends[:2], starts[1:3]))


def test_decode_over_ffmpeg_timeout_is_retried(make_site):
    site = one(make_site)
    site.behaviours, site.sleep = {"a": ["slow", "ok"]}, 4.0
    site.run(ffmpeg_timeout=1.0)
    assert site.read_jsonl("episodes.jsonl")[0]["reason"] is None
    assert len(site.decodes()) == 3 and len(site.decodes("end")) == 2


def test_failing_pass_2_aborts_the_run(make_site):
    site = one(make_site)
    site.behaviours = {"a": ["ok", "exit1"]}
    site.aborted()
    assert not site.completed() and not (site.out / "episodes.jsonl").exists()
    assert len(site.decodes()) == 2


def test_ffmpeg_stderr_is_printed_above_the_bar(make_site, capfd):
    """One tqdm.write per run: decoded with replacement, trailing newlines removed; the
    timed-out run's partial stderr too."""
    site = one(make_site)
    site.behaviours, site.sleep = {"a": ["slow", "ok"]}, 4.0
    site.noise = {"a": [b"first try\n", b"bad \xff byte\n\n"]}
    site.run(ffmpeg_timeout=1.0)
    shown = screen(capfd.readouterr().err)
    assert shown[1:3] == ["first try", "bad \N{REPLACEMENT CHARACTER} byte"]
    assert shown[0] == Bar("indices", 5, 5, "file") and shown[3].desc == "collect"


@pytest.mark.parametrize("workers", [1, 3])
def test_episodes_are_processed_on_up_to_workers_threads(site, workers):
    stems = [e.stem for e in site.world.eps.values()]
    site.behaviours, site.sleep = {s: ["slow", "slow"] for s in stems}, 0.15
    site.run(workers=workers)
    assert site.read_jsonl("clips.jsonl") == site.expected().clips
    events = sorted([(d["t"], 1) for d in site.decodes()]
                    + [(d["t"], -1) for d in site.decodes("end")])  # fmt: skip
    running = max(itertools.accumulate(e for _, e in events))
    assert running == 1 if workers == 1 else 2 <= running <= workers
