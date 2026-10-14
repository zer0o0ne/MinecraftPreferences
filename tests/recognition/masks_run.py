"""Runs of `main` of specs/gui-masks.md on the worlds of masks_world.py, and its stderr as the
spec fixes it: the bars, the `nothing to do` lines, the abort message (AGENTS.md Code, Progress)."""

import contextlib
import functools
import io
import json
from dataclasses import dataclass
from pathlib import Path

import masks_expected as me
import masks_world as mw
import pytest
from masks_files import write
from progress_bars import Bar, assert_aborted, tqdm_defaults

from mcpref.recognition.evaluate import main

SCENARIOS = {  # name: (relabel, config changes)
    "base": (False, {}),
    "relabelled": (True, {}),  # labels change, the split does not
    "one_b": (False, {"min_episodes": 6}),  # Q = beta, delta; (b) delta only; no (c)
    "unqualified": (False, {"min_episodes": 100}),  # no (b), no (c)
    "no_test": (False, {"split": ["0.5", "0.5", "0"]}),  # score test: nothing to do
    "nan": (False, {"min_episodes": 1}),  # gamma in Q, from one training episode
    "many_folds": (True, {"folds": 12}),  # empty folds; a fold without labelled frames
}


def config_of(name: str) -> dict:
    return {**mw.CONFIG, **SCENARIOS[name][1]}


def frames_of(name: str) -> list:
    return mw.frames(SCENARIOS[name][0])


@functools.cache
def expected_of(name: str) -> dict | None:
    """The oracle's run of scenario `name`; read-only."""
    return me.expected(frames_of(name), config_of(name))


UNITS = {"templates catalogue": "class", "templates b": "step", "templates c": "class"}
UNITS["features train"] = "frame"
UNITS.update(
    {"score validation": "frame", "thresholds": "point", "score test": "frame"}
)


def bars(exp: dict) -> list:
    """`screen` of a successful run, from the oracle's totals: each bar in run order at its
    total, or `<desc>: nothing to do` when its loop has no item (Interface)."""
    return [
        Bar(d, n, n, UNITS[d]) if n else f"{d}: nothing to do"
        for d, n in exp["totals"].items()
    ]


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


def run_main(
    root: Path, repo: Path, config: dict, fs: list, remove=(), catalogue=mw.CATALOGUE
) -> Run:
    """Writes a world of frames fs and `catalogue` under root, deletes the files `remove`
    (relative to root), runs main from another directory with tqdm's defaults (no TQDM_*
    variable)."""
    path = write(root, repo, fs, config, catalogue)
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
