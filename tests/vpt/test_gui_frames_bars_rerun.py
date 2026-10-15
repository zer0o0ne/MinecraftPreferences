"""Progress bars of `main run` in specs/vpt-gui-frames.md after an interruption in (a), in
the plan and in (c), and on a finished run: `indices` over the missing indices, `resume`
over the complete shards, `plan` from `initial`, `extract` over the frames without `.yuv`,
and the lines replacing them."""

import itertools
import shutil

from progress_bars import Bar, assert_aborted, assert_progress, parse, screen, states
from vpt_checks import key
from vpt_world import Ep

from mcpref.vpt.gui_frames import main

STORED = "indices: already done"


def plan_bar(site) -> Bar:
    rows, records = site.expected()
    return Bar("plan", len(rows), 6, "frame", f"{len(records)} episodes")


def extract_bar(total: int) -> Bar:
    return Bar("extract", total, total, "frame", "0 failed")


def drop(err: str, desc: str) -> str:
    """`err` without the lines drawn by the bar `desc`."""

    def drawn(line: str) -> bool:
        parts = [parse(s) for s in line.split("\r") if s.strip(" ")]
        return bool(parts) and all(b and b.desc == desc for b in parts)

    return "\n".join(line for line in err.split("\n") if not drawn(line))


def assert_resumed(err: str, initial: int) -> None:
    """`assert_progress` with the `plan` bar starting at `initial` instead of 0."""
    assert_progress(drop(err, "plan"))
    drawn = states(err, "plan")
    assert drawn[0].n == initial and len({(b.total, b.unit) for b in drawn}) == 1
    assert all(a.n <= b.n for a, b in itertools.pairwise(drawn))
    assert drawn[-1] in screen(err)


def test_bars_after_interruption_in_indices(site, capfd):
    """The fourth index fails; the rerun downloads the two left."""
    site.faults[site.get("/idx/9xx.json")] = [404]
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    site.faults.clear()
    assert main(["run", site.config()]) == 0
    err = capfd.readouterr().err
    indices = Bar("indices", 2, 2, "file")
    assert screen(err) == [
        indices,
        "resume: nothing to do",
        plan_bar(site),
        extract_bar(6),
    ]
    assert_progress(err)


def test_plan_bar_resumes_at_frames_of_complete_shards(site, capfd):
    """Stopped in draw 2 (K = 2): shard 0 holds 4 frames of 4 episodes; draw 2 takes 2
    frames of 1 episode. Per drawn episode: advanced by its frames, then the postfix."""
    site.stop_in_plan(capfd)
    assert main(["run", site.config()]) == 0
    err = capfd.readouterr().err
    resume = Bar("resume", 1, 1, "shard")
    assert screen(err) == [STORED, resume, plan_bar(site), extract_bar(6)]
    assert_resumed(err, 4)
    shown = {(b.n, b.postfix) for b in states(err, "plan")}
    after = {(4, "4 episodes"), (6, "5 episodes")}
    assert after <= shown <= after | {(6, "4 episodes"), (4, None)}


def test_bars_after_interruption_in_extraction(site, capfd):
    """A rename of shard 1 raises (a directory at `<id>.yuv`, removed before the rerun):
    the rerun extracts the frames still without `.yuv`."""
    rows, _ = site.expected()
    site.behaviours = {key(rows[4]): ["blocked"]}
    site.sleep = 0.0
    assert_aborted(main(["run", site.config()]), capfd.readouterr().err)
    shutil.rmtree(site.out / "frames" / f"{rows[4]['id']}.yuv")
    left = len(rows) - len([n for n in site.files() if n.endswith(".yuv")])
    assert left >= 1
    site.behaviours = {}
    assert main(["run", site.config()]) == 0
    err = capfd.readouterr().err
    resume = Bar("resume", 2, 2, "shard")
    assert screen(err) == [STORED, resume, "plan: already done", extract_bar(left)]
    assert_progress(err)


def test_extract_total_is_planned_frames_without_yuv(site, capfd):
    rows, _ = site.expected()
    site.behaviours = {key(rows[3]): ["exit1"]}
    assert main(["run", site.config(http_retries=0)]) == 0
    (site.out / "frames" / f"{rows[0]['id']}.yuv").unlink()
    assert states(capfd.readouterr().err, "extract")[-1].postfix == "1 failed"
    assert main(["run", site.config(http_retries=0)]) == 0
    err = capfd.readouterr().err
    resume = Bar("resume", 2, 2, "shard")
    assert screen(err) == [STORED, resume, "plan: already done", extract_bar(2)]


def test_finished_rerun_prints_already_done(site, capfd):
    path = site.config()
    assert main(["run", path]) == 0
    capfd.readouterr()
    assert main(["run", path]) == 0
    err = capfd.readouterr().err
    resume = Bar("resume", 2, 2, "shard")
    assert screen(err) == [
        STORED,
        resume,
        "plan: already done",
        "extract: already done",
    ]
    assert states(err, "extract") == [] and states(err, "plan") == []
    assert_progress(err)


def test_finished_empty_plan_rerun(make_site, capfd):
    """One shard of one draw without frames: `extract: nothing to do` again."""
    site = make_site(episodes={"6xx": [Ep("data/6.9/old6.mp4", reason="version")]})
    assert main(["run", site.config()]) == 0
    capfd.readouterr()
    assert main(["run", site.config()]) == 0
    err = capfd.readouterr().err
    resume = Bar("resume", 1, 1, "shard")
    assert screen(err) == [
        STORED,
        resume,
        "plan: already done",
        "extract: nothing to do",
    ]
