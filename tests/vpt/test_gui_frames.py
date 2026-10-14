"""`main run` of specs/vpt-gui-frames.md (Behaviour (c), Interface outputs) with a fake
HTTP server and a fake ffmpeg."""

import collections
import os
import subprocess
import sys

import pytest
from progress_bars import assert_aborted, screen
from vpt_checks import command_head, key, stem
from vpt_world import frame

from mcpref.vpt.gui_frames import main


@pytest.mark.parametrize("n", [6, 100])
def test_run_writes_planned_frames_and_records(site, n):
    """N = 6 cuts an episode short; N = 100 exhausts every pool."""
    code = main(["run", site.config(n_frames=n)])
    rows, records = site.expected(n_frames=n)
    assert code == 0
    assert site.read_jsonl("frames.jsonl") == rows
    assert site.read_jsonl("episodes.jsonl") == records
    assert site.files() == {f"{r['id']}.yuv": frame(stem(r), r["frame"]) for r in rows}


def test_run_json_holds_config_etags_and_counts(site):
    site.etags["8xx"] = None
    path = site.config(n_frames=100)
    assert main(["run", path]) == 0
    _, records = site.expected(n_frames=100)
    reasons = collections.Counter(r["reason"] or "null" for r in records)
    assert site.run_json() == {
        "config": site.fields | {"n_frames": 100},
        "index_etags": site.etags,
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
    rows, _ = site.expected()
    site.behaviours = {key(r): ["slow"] for r in rows}
    site.sleep = 0.5
    assert main(["run", site.config(workers=3)]) == 0
    events = sorted(
        [(c["t"], 1) for c in site.calls()] + [(c["t"], -1) for c in site.calls("end")]
    )
    running = [sum(d for _, d in events[: k + 1]) for k in range(len(events))]
    assert max(running) == 3


def test_frames_jsonl_in_plan_order_when_extraction_finishes_out_of_order(site):
    rows, _ = site.expected()
    site.behaviours = {key(rows[0]): ["slow"]}
    site.sleep = 1.0
    assert main(["run", site.config()]) == 0
    assert site.calls("end")[-1]["key"] == key(rows[0])
    assert site.read_jsonl("frames.jsonl") == rows


FAILURES = ["exit1", "short", "long", "slow"]  # slow: longer than ffmpeg_timeout


@pytest.mark.parametrize("failure", FAILURES)
def test_failed_extraction_is_retried_after_delay(site, failure):
    rows, _ = site.expected()
    failing = key(rows[2])
    site.behaviours = {failing: [failure, "ok"]}
    assert main(["run", site.config(ffmpeg_timeout=1.0, retry_delay=0.2)]) == 0
    starts = [c["t"] for c in site.calls() if c["key"] == failing]
    assert len(starts) == 2 and starts[1] - starts[0] >= 0.2
    assert site.read_jsonl("frames.jsonl") == rows
    assert site.files()[f"{rows[2]['id']}.yuv"] == frame(
        stem(rows[2]), rows[2]["frame"]
    )


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("retries", [0, 2])
def test_frame_failed_after_retries_is_counted_not_fatal(site, capfd, failure, retries):
    rows, _ = site.expected()
    failing = key(rows[2])
    site.behaviours = {failing: [failure] * (retries + 1)}
    cfg = site.config(ffmpeg_timeout=1.0, http_retries=retries, retry_delay=0.05)
    assert main(["run", cfg]) == 0
    assert len([c for c in site.calls() if c["key"] == failing]) == retries + 1
    assert site.read_jsonl("frames.jsonl") == [
        r | {"status": "failed"} if r is rows[2] else r for r in rows
    ]
    assert f"{rows[2]['id']}.yuv" not in site.files()
    assert site.run_json()["frames"] == {"ok": 5, "failed": 1}


def test_no_part_file_left_for_extracted_frames(site):
    assert main(["run", site.config()]) == 0
    assert not [name for name in site.files() if name.endswith(".part")]


@pytest.mark.parametrize("ok", [True, False], ids=["ok", "abort"])
def test_module_entry_point(site, ok):
    """`python -m mcpref.vpt.gui_frames run <config>`; an index 404 aborts. Bars are drawn
    on stderr also when it is a pipe (Library mapping: disable=False); stderr is read as
    bytes, universal newlines would turn its carriage returns into line ends."""
    if not ok:
        site.faults["/idx/9xx.json"] = [404]
    cmd = [sys.executable, "-m", "mcpref.vpt.gui_frames", "run", site.config()]
    proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    err = proc.stderr.decode()
    if ok:
        assert proc.returncode == 0, err
        assert site.read_jsonl("frames.jsonl") == site.expected()[0]
        assert proc.stdout == b""
        shown = [(b.desc, b.n, b.total) for b in screen(err)]
        assert shown == [("indices", 5, 5), ("plan", 6, 6), ("extract", 6, 6)]
    else:
        assert_aborted(proc.returncode, err)
