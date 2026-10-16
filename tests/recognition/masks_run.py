"""Runs of `main` of specs/gui-masks.md on the worlds of masks_world.py, its stderr as the spec
fixes it (the bars and their lines, the abort message; AGENTS.md Code, Progress), a fake
raising inside a bar, and readers of its outputs."""

import contextlib
import functools
import io
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import masks_expected as me
import masks_world as mw
import numpy as np
import pytest
import tqdm
from masks_files import write
from progress_bars import Bar, assert_aborted, states, tqdm_defaults

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
UNITS = {"index": "frame", "templates catalogue": "class", "templates b": "step"}
UNITS.update({"templates c": "class", "model": "shard", "features train": "frame"})
UNITS.update(
    {"score validation": "frame", "thresholds": "frame", "score test": "frame"}
)
UNITS["report"] = "frame"
SETS = ("index", "templates catalogue", "templates b", "templates c")
SETS += ("features train", "score validation", "score test")


def config_of(name: str) -> dict:
    return {**mw.CONFIG, **SCENARIOS[name][1]}


def frames_of(name: str) -> list:
    return mw.frames(SCENARIOS[name][0])


@functools.cache
def expected_of(name: str) -> dict | None:
    """The oracle's run of scenario `name`; read-only."""
    return me.expected(frames_of(name), config_of(name))


def line(desc: str, items: list, h: int) -> Bar | str:
    """A stored set's bar at its total, or its line (Interface, progress bars); h: its shards
    complete before the run."""
    if not items or (h < len(items) and not sum(items[h:])):
        return f"{desc}: nothing to do"
    if h == len(items):
        return f"{desc}: already done"
    return Bar(desc, sum(items), sum(items), UNITS[desc])


def bars(exp: dict, done: dict | None = None, thresholds: bool = False) -> list:
    """`screen` of a run that exits 0: the bars in run order; done: desc -> the shards of its
    set complete before the run; thresholds: thresholds.json present before the run."""
    done, sets, out = done or {}, exp["items"], []
    for desc, unit in UNITS.items():
        if desc in sets:
            out.append(line(desc, sets[desc], done.get(desc, 0)))
        elif desc == "model":
            n = sum(len(sets[d]) for d in SETS[1:4])
            out.append(Bar(desc, n, n, unit))
        elif desc == "thresholds" and thresholds:
            out.append("thresholds: already done")
        else:
            n = len(exp["frames"]["validation" if desc == "thresholds" else "test"])
            out.append(Bar(desc, n, n, unit) if n else f"{desc}: nothing to do")
    return out


def initial(exp: dict, done: dict) -> dict:
    """desc -> `initial` of the bar of each stored set: the items of its complete shards."""
    return {d: sum(exp["items"][d][:h]) for d, h in done.items()}


def assert_bars(err: str, start: dict) -> None:
    """The bars of a continued run (`assert_progress` with `initial`): each drawn from
    start[desc] (0 if absent), never going back, keeping its total and unit."""
    for desc in dict.fromkeys(b.desc for b in states(err)):
        drawn = states(err, desc)
        assert drawn[0].n == start.get(desc, 0), drawn
        assert len({(b.total, b.unit) for b in drawn}) == 1, drawn
        assert all(a.n <= b.n for a, b in itertools.pairwise(drawn)), drawn


def finished(exp: dict) -> dict:
    """`done` of a run after a complete one: every set complete."""
    return {desc: len(exp["items"][desc]) for desc in SETS}


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


class Interrupted(RuntimeError):
    """The fake's failure inside a bar."""


def interrupting(desc: str, at: int):
    """A patch making tqdm's `update` raise once the bar `desc` reaches `at`."""
    real = tqdm.tqdm.update

    def update(self, n=1):
        shown = real(self, n)
        if self.desc == desc and self.n >= at:
            raise Interrupted("interrupted by the test")
        return shown

    return lambda mp: mp.setattr(tqdm.tqdm, "update", update)


def rerun(root: Path, patch=None) -> Run:
    """Runs main on the world under root from another directory, with tqdm's defaults (no
    TQDM_* variable) and `patch(monkeypatch)` applied."""
    (root / "cwd").mkdir(exist_ok=True)
    err, out = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        tqdm_defaults(mp)
        mp.chdir(root / "cwd")
        if patch:
            patch(mp)
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            code = main([str(root / "gui-masks.yaml")])
    report = root / "out/report.json"
    report = json.loads(report.read_text()) if report.exists() else None
    return Run(code, err.getvalue(), out.getvalue(), root / "out", report)


def run_main(
    root: Path, repo: Path, config: dict, fs: list, remove=(), catalogue=mw.CATALOGUE
) -> Run:
    """Writes a world of frames fs and `catalogue` under root, deletes the files `remove`
    (relative to root), runs main (`rerun`)."""
    write(root, repo, fs, config, catalogue)
    for name in remove:
        (root / name).unlink()
    return rerun(root)


def npz(path: Path) -> dict:
    """A shard's .npz, loaded without pickle (no object arrays)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def shard_files(directory: Path, file: str) -> list:
    """The paths of `file` in the shards 000000, 000001, ... of a set, in shard order."""
    names = sorted(p.name for p in directory.iterdir()) if directory.exists() else []
    assert names == [f"{k:06d}" for k in range(len(names))], names
    return [directory / n / file for n in names]


def tree(out: Path) -> dict:
    """Every file under out: path relative to out -> its bytes."""
    return {
        p.relative_to(out).as_posix(): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
