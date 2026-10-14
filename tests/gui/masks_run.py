"""Runs of `main` of specs/gui-masks.md on the worlds of masks_world.py, and its stderr as the
spec fixes it: the bars, the `nothing to do` lines, the abort message (AGENTS.md Code, Progress)."""

import contextlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import masks_world as mw
import pytest
from progress_bars import Bar, assert_aborted, tqdm_defaults

from mcpref.gui.evaluate import main


def bars(exp: dict, config: dict) -> list:
    """`screen` of a successful run, from the oracle's report `exp`: each bar in run order at its
    total, or `<desc>: nothing to do` when its loop has no item (Interface)."""
    tr, va, te = exp["split"]  # frames per label of the frames built from or scored
    rows = math.ceil(mw.H / config["chunk_rows"])
    grid = [len(config[g]) for g in ("tau_grid", "theta_grid", "mu_grid")]
    loops = [
        ("templates b", "step", sum(tr[k] + rows for k in exp["qualified"])),
        ("templates c", "class", len(exp["c"]["pairs"])),
        ("score validation", "frame", 3 * sum(va.values())),
        ("thresholds", "point", 3 * math.prod(grid)),
        ("score test", "frame", 3 * sum(te.values())),
    ]
    return [Bar(d, n, n, u) if n else f"{d}: nothing to do" for d, u, n in loops]


@dataclass
class Run:
    code: int
    err: str
    stdout: str
    out: Path
    report: dict | None

    def failed(self) -> list:
        """`assert_aborted` (Interface): the lines above the message, as `screen` shows them."""
        return assert_aborted(self.code, self.err)


def run_main(root: Path, repo: Path, config: dict, relabel=False, remove=()) -> Run:
    """Writes a world under root, deletes the files `remove` (relative to root), runs main from
    another directory with tqdm's defaults (no TQDM_* variable)."""
    path = mw.write(root, repo, mw.frames(relabel), config)
    for name in remove:
        (root / name).unlink()
    (root / "cwd").mkdir()
    err, out = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        tqdm_defaults(mp)
        mp.chdir(root / "cwd")
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = main([str(path)])
    report = root / "out/report.json"
    report = json.loads(report.read_text()) if report.exists() else None
    return Run(code, err.getvalue(), out.getvalue(), root / "out", report)
