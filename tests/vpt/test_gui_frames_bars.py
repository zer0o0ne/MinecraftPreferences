"""Progress bars of `main run` in specs/vpt-gui-frames.md (Interface, AGENTS.md Progress):
`indices`, `plan`, `extract`, the lines replacing them, ffmpeg's stderr above them, the
abort message below them; read off stderr in tqdm's default format."""

import itertools
import re

import pytest
from progress_bars import Bar, assert_aborted, assert_progress, screen, states
from vpt_checks import key
from vpt_world import Ep

from mcpref.vpt.gui_frames import main

INDICES = Bar("indices", 5, 5, "file")


def plan_bar(site, n: int = 6) -> Bar:
    """Final `plan` state for N = n: frames planned, postfix with the episodes drawn."""
    rows, records = site.expected(n_frames=n)
    return Bar("plan", len(rows), n, "frame", f"{len(records)} episodes")


@pytest.mark.parametrize("n", [6, 100])
def test_run_bars_totals_and_final_counts(site, capfd, n):
    """N = 6 cuts the plan at N; N = 100 ends below N, the pools run dry."""
    code = main(["run", site.config(n_frames=n)])
    out, err = capfd.readouterr()
    rows, _ = site.expected(n_frames=n)
    assert code == 0 and out == ""
    extract = Bar("extract", len(rows), len(rows), "frame", "0 failed")
    assert screen(err) == [INDICES, plan_bar(site, n), extract]
    assert_progress(err)


@pytest.mark.parametrize("n", [6, 100])
def test_plan_bar_advances_by_each_drawn_episodes_frames(site, capfd, n):
    """Per drawn episode: advanced by its frames (0 if none), then postfix `<k> episodes`."""
    assert main(["run", site.config(n_frames=n)]) == 0
    rows, records = site.expected(n_frames=n)
    taken = [sum(r["relpath"] == e["relpath"] for r in rows) for e in records]
    planned = list(itertools.accumulate(taken, initial=0))
    after = {(planned[k], f"{k} episodes") for k in range(len(records) + 1)}
    between = {(planned[k + 1], f"{k} episodes") for k in range(len(records))}
    shown = states(capfd.readouterr().err, "plan")
    assert after <= {(b.n, b.postfix) for b in shown} <= after | between | {(0, None)}
    assert {b.total for b in shown} == {n}


def test_extract_bar_counts_each_result_and_its_failures(site, capfd):
    rows, _ = site.expected()
    site.behaviours = {key(rows[1]): ["exit1"], key(rows[4]): ["short"]}
    assert main(["run", site.config(http_retries=0)]) == 0
    shown = states(capfd.readouterr().err, "extract")
    assert shown[-1] == Bar("extract", 6, 6, "frame", "2 failed")
    assert all(b.postfix or b.n == 0 for b in shown)
    counted = [
        (b.n, int(b.postfix.removesuffix(" failed"))) for b in shown if b.postfix
    ]
    assert all(re.fullmatch(r"\d+ failed", b.postfix) for b in shown if b.postfix)
    assert {n for n, _ in counted} == set(range(7)) and all(f <= n for n, f in counted)
    assert all(a <= c and b <= d for (a, b), (c, d) in itertools.pairwise(counted))


def test_extract_total_is_planned_frames_without_yuv(site, capfd):
    rows, _ = site.expected()
    site.behaviours = {key(rows[3]): ["exit1"]}
    assert main(["run", site.config(http_retries=0)]) == 0
    (site.out / "frames" / f"{rows[0]['id']}.yuv").unlink()
    assert states(capfd.readouterr().err, "extract")[-1].postfix == "1 failed"
    assert main(["run", site.config(http_retries=0)]) == 0
    err = capfd.readouterr().err
    assert screen(err) == [
        INDICES,
        plan_bar(site),
        Bar("extract", 2, 2, "frame", "0 failed"),
    ]


def test_rerun_with_every_frame_extracted_prints_already_done(site, capfd):
    path = site.config()
    assert main(["run", path]) == 0
    capfd.readouterr()
    assert main(["run", path]) == 0
    err = capfd.readouterr().err
    assert screen(err) == [INDICES, plan_bar(site), "extract: already done"]
    assert states(err, "extract") == []
    assert_progress(err)


def test_empty_plan_prints_nothing_to_do(make_site, capfd):
    site = make_site(episodes={"6xx": [Ep("data/6.9/old6.mp4", reason="version")]})
    assert main(["run", site.config()]) == 0
    err = capfd.readouterr().err
    assert screen(err) == [INDICES, plan_bar(site), "extract: nothing to do"]
    assert plan_bar(site) == Bar("plan", 0, 6, "frame", "1 episodes")
    assert states(err, "extract") == []
    assert_progress(err)


def test_abort_while_downloading_indices_follows_the_closed_bar(site, capfd):
    """9xx, the fourth index, fails: three downloads advanced the bar."""
    site.faults["/idx/9xx.json"] = [404]
    code = main(["run", site.config()])
    err = capfd.readouterr().err
    assert_aborted(code, err)
    assert screen(err)[:-1] == [Bar("indices", 3, 5, "file")]


@pytest.mark.parametrize("suffix", ["-options.json", ".jsonl"])
def test_abort_while_planning_follows_the_closed_bars(site, capfd, suffix):
    """The first downloaded episode fails; only `version` episodes precede it."""
    site.faults[site.first_fetched(suffix)] = [503] * 10
    code = main(["run", site.config(http_retries=0)])
    err = capfd.readouterr().err
    assert_aborted(code, err)
    shown = [(b.desc, b.n, b.total, b.unit) for b in screen(err)[:-1]]
    assert shown == [("indices", 5, 5, "file"), ("plan", 0, 6, "frame")]


def test_raising_result_cancels_queued_frames_then_closes_the_bar(site, capfd):
    """Every rename raises; one worker: the first result raises while at most one more
    frame has started; it finishes, the four or more still queued never start."""
    rows, _ = site.expected()
    site.behaviours = {key(r): ["blocked"] for r in rows}
    site.sleep = 0.5
    code = main(["run", site.config(workers=1)])
    err = capfd.readouterr().err
    assert_aborted(code, err)
    assert 1 <= len(site.calls()) <= 2 and len(site.calls("end")) == len(site.calls())
    extract = Bar("extract", 0, 6, "frame", "0 failed")
    assert screen(err)[:-1] == [INDICES, plan_bar(site), extract]


@pytest.mark.parametrize("failure", ["exit1", "slow"])  # slow: killed at ffmpeg_timeout
def test_ffmpeg_stderr_printed_after_each_run_above_the_bar(site, capfd, failure):
    """Run 1 fails, its (partial) stderr is printed; run 2 succeeds, its stderr is decoded
    with replacement and loses its trailing newlines."""
    rows, _ = site.expected()
    site.behaviours = {key(rows[1]): [failure, "ok"]}
    site.noise = {key(rows[1]): [b"first try\n", b"bad \xff byte\n\n"]}
    assert main(["run", site.config(ffmpeg_timeout=1.0, retry_delay=0.05)]) == 0
    extract = Bar("extract", 6, 6, "frame", "0 failed")
    shown = [INDICES, plan_bar(site), "first try", "bad � byte", extract]
    err = capfd.readouterr().err
    assert screen(err) == shown
    assert_progress(err)


def test_large_ffmpeg_stderr_never_blocks_extraction(site, capfd):
    """256 KiB of stderr, beyond a pipe's buffer, before the frame is written."""
    rows, _ = site.expected()
    big = "e" * 2**18
    site.noise = {key(rows[0]): [big.encode()]}
    assert main(["run", site.config(ffmpeg_timeout=2.0, http_retries=0)]) == 0
    assert site.read_jsonl("frames.jsonl") == rows
    assert screen(capfd.readouterr().err).count(big) == 1
