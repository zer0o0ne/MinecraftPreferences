"""src/mcpref/recognition/evaluate.py against specs/gui-masks.md: config, split, the worlds of
masks_world.py, the bars on stderr, running as a module; its outputs in
test_evaluate_outputs.py, its aborts in test_evaluate_aborts.py."""

import dataclasses
import subprocess
import sys
from fractions import Fraction

import masks_expected as me
import masks_world as mw
import numpy as np
import pytest
import yaml
from masks_files import write
from masks_run import SCENARIOS, Run, bars, config_of, expected_of
from progress_bars import Bar, assert_progress, screen

from mcpref.recognition.evaluate import GuiMasksConfig, load_config

PATHS = {"frames_dir": "data/gui_frames", "labels_dir": "data/gui_labels"}
PATHS.update(screens="configs/gui-screens.yaml", out_dir="data/gui_masks")
NUMBERS = {"seed": 0, "min_episodes": 10, "spread_quantile": 0.9, "spread_max": 24}
NUMBERS.update(patch=8, patches_per_class=64, eps=1, chunk_rows=40)
NUMBERS.update(folds=5, fold_seed=1, feature_quantile=0.75)
GRIDS = {"tau_grid": [8, 16, 24, 32, 48]}
GRIDS["theta_grid"] = [Fraction(t) for t in ("0.5", "0.6", "0.7", "0.8", "0.9", "0.95")]
GRIDS["mu_grid"] = [Fraction(m) for m in ("0", "0.05", "0.1", "0.2")]
SPLIT = [Fraction(3, 5), Fraction(1, 5), Fraction(1, 5)]


def test_load_config_starting_values(repo, tmp_path, monkeypatch):
    """Starting values; paths relative to the repository root, whatever the working directory;
    theta and mu grids exact fractions."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(repo / "configs/gui-masks.yaml")
    assert isinstance(cfg, GuiMasksConfig)
    fields = {f.name for f in dataclasses.fields(cfg)}
    assert fields == {*PATHS, *NUMBERS, *GRIDS, "split"}
    assert {k: getattr(cfg, k) for k in NUMBERS} == NUMBERS
    assert {k: list(getattr(cfg, k)) for k in GRIDS} == GRIDS
    assert [Fraction(str(p)) for p in cfg.split] == SPLIT
    paths = {k: (repo / getattr(cfg, k)).resolve() for k in PATHS}
    assert paths == {k: (repo / v).resolve() for k, v in PATHS.items()}
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


@pytest.fixture
def config_with(repo, tmp_path):
    """Loads the starting config with `changes`."""

    def load(**changes) -> GuiMasksConfig:
        cfg = yaml.safe_load((repo / "configs/gui-masks.yaml").read_text())
        (tmp_path / "gui-masks.yaml").write_text(yaml.safe_dump({**cfg, **changes}))
        return load_config(tmp_path / "gui-masks.yaml")

    return load


BAD = [  # Notation: violated constraints
    {"split": ["0.5", "0.3", "0.3"]},
    {"split": ["0.5", "0.5"]},
    {"split": ["0.5", "0.25", "0.25000000000000001"]},  # sums to 1 only in floats
    *({"patch": v} for v in (3, 16)),
    {"patches_per_class": 0},
    *({"min_episodes": v} for v in (0, -1, 2.0, True)),  # an integer >= 1
    *({"eps": v} for v in (0, -1)),
    *({"chunk_rows": v} for v in (0, 1, 3, -2, True)),  # even, >= 2
    *({"folds": v} for v in (1, 2.5)),
    {"fold_seed": 1.5},
    *({"feature_quantile": v} for v in (0, -0.25, 1.25)),
]
GOOD = [  # boundaries inside the constraints
    {"split": ["0.7", "0.2", "0.1"]},  # 0.7 + 0.2 + 0.1 < 1 in floats
    {"split": ["1", "0", "0"]},
    *({"patch": v} for v in (40, 1)),
    {"patches_per_class": 1},
    {"min_episodes": 1},
    {"eps": 0.001},
    {"chunk_rows": 2},
    {"folds": 2},
    {"fold_seed": 0},
    *({"feature_quantile": v} for v in (1, 0.001)),
]


@pytest.mark.parametrize("changes", BAD)
def test_load_config_rejects(config_with, changes):
    with pytest.raises(ValueError):
        config_with(**changes)


@pytest.mark.parametrize("changes", GOOD)
def test_load_config_accepts(config_with, changes):
    assert isinstance(config_with(**changes), GuiMasksConfig)


def test_worlds_exercise_the_split():
    """Counting failed records or only labelled episodes would change the split; validation
    and test episodes hold invalid and no_answer frames, which are not scored."""
    fs, part = mw.frames(), expected_of("base")["part"]
    records = me.parts([dataclasses.replace(f, status="ok") for f in fs], mw.CONFIG)
    assert {e: records[e] for e in part} != part
    relabelled = [f for f in mw.frames(True) if f.label not in (None, *me.BAD)]
    labelled = me.parts(relabelled, mw.CONFIG)
    assert any(labelled[e] != part[e] for e in labelled)
    scored = [f for f in fs if f.status == "ok" and part[f.relpath] > 0]
    assert set(me.BAD) <= {f.label for f in scored}


def classes(exp: dict, v: str) -> list:
    return list(exp["pairs"][v])


def test_worlds_exercise_the_variants():
    """beta has no (b) template, (c) is built for alpha and delta only; gamma (in F^te) has a
    template in (d) only; (d) skips alpha's member without a label; (e) skips gamma (slot over
    a transparent canvas) and has no pair for the classes without slots; in `one_b` a single
    (b) template and no (c)."""
    exp = expected_of("base")
    assert exp["qualified"] == ["alpha", "beta", "delta"]
    assert classes(exp, "b") == classes(exp, "c") == ["alpha", "delta"]
    assert exp["a"]["agreement"]["test"]["per_label"]["gamma"]["n_true"] > 0
    assert [v for v in mw.VARIANTS if "gamma" in exp["pairs"][v]] == ["d"]
    assert list(exp["pairs"]["d"]["alpha"]) == ["__one"]
    assert classes(exp, "e") == ["delta"]
    assert list(exp["pairs"]["f"]) == ["alpha", "beta", "delta"]
    one_b = expected_of("one_b")
    assert classes(one_b, "b") == ["delta"] and classes(one_b, "c") == []


def test_worlds_exercise_the_folds():
    """(13), (14): in `base` beta has fold (b) templates but none from all of F^tr, so the
    fold (c) leaves it out; in `one_b` folds have two (b) templates but no (c), as all of F^tr
    has one; `nan` puts gamma in Q from one training episode, so its fold has no gamma pair
    (NaN) and builds (c) over two classes; `many_folds` has empty folds and a fold whose
    episode has no labelled frame."""
    base = expected_of("base")["fold_classes"]
    assert all(
        b == ["alpha", "beta", "delta"] and c == ["alpha", "delta"]
        for b, c in base.values()
    )
    one_b = expected_of("one_b")["fold_classes"]
    assert all(b == ["beta", "delta"] and c == [] for b, c in one_b.values())
    nan = expected_of("nan")
    x = np.array(nan["x"]["train"])
    columns = [c for i, c in enumerate(nan["columns"]) if np.isnan(x[:, i]).any()]
    assert columns == ["b/gamma", "c/gamma"]
    assert ["alpha", "delta"] in [c for _, c in nan["fold_classes"].values()]
    many = expected_of("many_folds")["folds"]
    assert (0, 0) in many and any(n > 0 and frames == 0 for n, frames in many)


def test_worlds_exercise_the_bars():
    """Each optional bar is drawn in some world and replaced by its line in another."""
    lines = {line for name in SCENARIOS for line in bars(expected_of(name))}
    loops = ("templates b", "templates c", "score test", "report")
    assert {f"{d}: nothing to do" for d in loops} <= lines
    assert set(loops) <= {line.desc for line in lines if isinstance(line, Bar)}


def test_bars_on_stderr(world):
    """Interface, progress bars: stderr holds only, in run order, `index` (unit frame, the
    `ok` count), `templates catalogue` (class, |K|), `templates b` (step, |Q| ceil(H /
    chunk_rows)), `templates c` (class, the classes taking part summed over the builds),
    `model` (shard, the template shards), `features train`, `score validation`, `thresholds`,
    `score test`, `report` (frame, the part's frames), each from 0 to its total without
    postfix, or `<desc>: nothing to do`; nothing on stdout."""
    _, run, exp = world
    assert run.code == 0, run.err
    assert screen(run.err) == bars(exp) and run.stdout == ""
    assert_progress(run.err)


def test_runs_as_module(tmp_path, repo):
    """`python -m mcpref.recognition.evaluate <config path>`, on an empty validation set:
    the `index` bar, then the message."""
    config = {**config_of("base"), "split": ["0.5", "0", "0.5"]}
    path = write(tmp_path, repo, mw.frames(), config)
    argv = [sys.executable, "-m", "mcpref.recognition.evaluate", str(path)]
    proc = subprocess.run(
        argv, capture_output=True, timeout=8, check=False, cwd=tmp_path
    )
    err = proc.stderr.decode()  # bytes: text mode would turn tqdm's '\r' into '\n'
    run = Run(proc.returncode, err, proc.stdout.decode(), tmp_path / "out", None)
    assert run.failed() == [Bar("index", 54, 54, "frame")]
    assert err.endswith(
        "mcpref.recognition.evaluate: RuntimeError: the validation part is empty\n"
    )
