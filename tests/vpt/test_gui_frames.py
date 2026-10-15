"""`main run` of specs/vpt-gui-frames.md (Behaviour (a)-(c), Interface outputs) with a fake
HTTP server and a fake ffmpeg."""

import collections
import itertools
import os
import subprocess
import sys

import pytest
from progress_bars import assert_aborted, screen
from vpt_checks import command_head, key, stem
from vpt_world import frame

from mcpref.vpt.gui_frames import main


@pytest.mark.parametrize("n", [6, 100])
def test_run_writes_indices_plan_shards_and_frames(site, n):
    """N = 6 cuts an episode short; N = 100 exhausts every pool."""
    assert main(["run", site.config(n_frames=n)]) == 0
    assert site.plan() == site.shards(n_frames=n)
    rows, _ = site.expected(n_frames=n)
    assert site.files() == {f"{r['id']}.yuv": frame(stem(r), r["frame"]) for r in rows}
    indices = {p.name: p.read_bytes() for p in (site.out / "indices").iterdir()}
    assert indices == {f"{s}.json": site.world.index(s) for s in site.world.index_urls}
    assert sorted(os.listdir(site.out)) == ["frames", "indices", "plan", "run.json"]


def test_run_json_holds_config_and_counts(site):
    path = site.config(n_frames=100)
    assert main(["run", path]) == 0
    _, records = site.expected(n_frames=100)
    reasons = collections.Counter(r["reason"] or "null" for r in records)
    assert site.run_json() == {
        "config": site.fields | {"n_frames": 100},
        "episodes": dict(reasons),
        "frames": {"ok": 10},
    }


def test_ffmpeg_gets_the_interface_command(site):
    assert main(["run", site.config(ffmpeg_rw_timeout=11)]) == 0
    rows, _ = site.expected()
    calls = sorted(site.calls(), key=lambda c: c["key"])
    for call, row in zip(calls, sorted(rows, key=key)):
        url = site.world.basedirs[row["subset"]] + row["relpath"]
        head = command_head(site.fields | {"ffmpeg_rw_timeout": 11}, url, row["frame"])
        part = site.out / "frames" / f"{row['id']}.yuv.part"
        assert call["argv"][:-1] == head
        assert os.path.realpath(call["argv"][-1]) == os.path.realpath(part)
    assert len(calls) == len(rows)


def test_extraction_runs_on_workers_threads(site):
    """K = 3: the six frames in one shard."""
    rows, _ = site.expected()
    site.behaviours = {key(r): ["slow"] for r in rows}
    site.sleep = 0.5
    assert main(["run", site.config(workers=3, plan_shard_draws=3)]) == 0
    events = sorted(
        [(c["t"], 1) for c in site.calls()] + [(c["t"], -1) for c in site.calls("end")]
    )
    running = [sum(d for _, d in events[: k + 1]) for k in range(len(events))]
    assert max(running) == 3


def test_next_shard_extracted_after_every_result_of_a_shard(site):
    """K = 1: shards of two frames each. In shard 0 one frame is slow, the other fails (no
    retries); in shard 1 one is slow: no frame of a shard starts before every run of the
    previous shard has ended."""
    by_shard = [[key(f) for f in s["frames"]] for s in site.shards(plan_shard_draws=1)]
    assert [len(keys) for keys in by_shard] == [2, 2, 2]
    site.behaviours = {by_shard[0][0]: ["slow"], by_shard[0][1]: ["exit1"],
                       by_shard[1][1]: ["slow"]}  # fmt: skip
    site.sleep = 0.6
    cfg = site.config(workers=3, plan_shard_draws=1, http_retries=0)
    assert main(["run", cfg]) == 0
    for done, later in itertools.pairwise(by_shard):
        ended = max(c["t"] for c in site.calls("end") if c["key"] in done)
        assert min(c["t"] for c in site.calls() if c["key"] in later) > ended
    assert site.run_json()["frames"] == {"ok": 5, "failed": 1}


FAILURES = ["exit1", "short", "long", "slow"]  # slow: longer than ffmpeg_timeout


@pytest.mark.parametrize("failure", FAILURES)
def test_failed_extraction_is_retried_after_delay(site, failure):
    rows, _ = site.expected()
    failing = key(rows[2])
    site.behaviours = {failing: [failure, "ok"]}
    assert main(["run", site.config(ffmpeg_timeout=1.0, retry_delay=0.2)]) == 0
    starts = [c["t"] for c in site.calls() if c["key"] == failing]
    assert len(starts) == 2 and starts[1] - starts[0] >= 0.2
    assert site.files()[f"{rows[2]['id']}.yuv"] == frame(
        stem(rows[2]), rows[2]["frame"]
    )
    assert site.run_json()["frames"] == {"ok": 6}


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("retries", [0, 2])
def test_frame_failed_after_retries_is_counted_not_fatal(site, failure, retries):
    """Its status is `failed`: no `<id>.yuv`; the plan shards are unchanged."""
    rows, _ = site.expected()
    failing = key(rows[2])
    site.behaviours = {failing: [failure] * (retries + 1)}
    cfg = site.config(ffmpeg_timeout=1.0, http_retries=retries, retry_delay=0.05)
    assert main(["run", cfg]) == 0
    assert len([c for c in site.calls() if c["key"] == failing]) == retries + 1
    yuv = {name for name in site.files() if name.endswith(".yuv")}
    assert yuv == {f"{r['id']}.yuv" for r in rows if r is not rows[2]}
    assert site.plan() == site.shards()
    assert site.run_json()["frames"] == {"ok": 5, "failed": 1}


def test_every_frame_failed_counts_only_failed(site):
    """run.json counts only statuses that occurred."""
    site.behaviours = {key(r): ["exit1"] for r in site.expected()[0]}
    assert main(["run", site.config(http_retries=0)]) == 0
    assert site.run_json()["frames"] == {"failed": 6}


def test_no_part_file_left_for_extracted_frames(site):
    assert main(["run", site.config()]) == 0
    assert not [name for name in site.files() if name.endswith(".part")]


@pytest.mark.parametrize("ok", [True, False], ids=["ok", "abort"])
def test_module_entry_point(site, ok):
    """`python -m mcpref.vpt.gui_frames run <config>`; an index 404 aborts. Bars are drawn
    on stderr also when it is a pipe (Library mapping: disable=False); stderr is read as
    bytes, universal newlines would turn its carriage returns into line ends."""
    if not ok:
        site.faults[site.get("/idx/9xx.json")] = [404]
    cmd = [sys.executable, "-m", "mcpref.vpt.gui_frames", "run", site.config()]
    proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    err = proc.stderr.decode()
    if ok:
        assert proc.returncode == 0, err
        assert site.plan() == site.shards()
        assert proc.stdout == b""
        shown = [(b.desc, b.n, b.total) if not isinstance(b, str) else b
                 for b in screen(err)]  # fmt: skip
        assert shown == [("indices", 5, 5), "resume: nothing to do", ("plan", 6, 6),
                         ("extract", 6, 6)]  # fmt: skip
    else:
        assert_aborted(proc.returncode, err)
