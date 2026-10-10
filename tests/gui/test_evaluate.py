"""src/mcpref/gui/evaluate.py against specs/gui-masks.md: config, split, and `main` end to end
on the synthetic worlds of masks_world.py (report.json against masks_oracle.py, templates,
report.md, progress, exits)."""

import contextlib
import dataclasses
import io
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import masks_world as mw
import numpy as np
import pytest
import yaml
from PIL import Image

from mcpref.gui.evaluate import GuiMasksConfig, load_config, main, split_episodes

PATHS = {"frames_dir": "data/gui_frames", "labels_dir": "data/gui_labels"}
PATHS.update(screens="configs/gui-screens.yaml", out_dir="data/gui_masks")
NUMBERS = {"seed": 0, "min_episodes": 10, "spread_quantile": 0.9, "spread_max": 24}
NUMBERS.update(patch=8, patches_per_class=64, eps=1, chunk_rows=40)
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
    {"patch": 3},
    {"patch": 16},
    {"patches_per_class": 0},
    {"eps": 0},
    {"eps": -1},
]
GOOD = [  # boundaries inside the constraints
    {"split": ["0.7", "0.2", "0.1"]},  # 0.7 + 0.2 + 0.1 < 1 in floats
    {"split": ["1", "0", "0"]},
    {"patch": 40},
    {"patch": 1},
    {"patches_per_class": 1},
    {"eps": 0.001},
]


@pytest.mark.parametrize("changes", BAD)
def test_load_config_rejects(config_with, changes):
    with pytest.raises(ValueError):
        config_with(**changes)


@pytest.mark.parametrize("changes", GOOD)
def test_load_config_accepts(config_with, changes):
    assert isinstance(config_with(**changes), GuiMasksConfig)


@pytest.mark.parametrize("seed", [0, 1, 230])
@pytest.mark.parametrize("episodes", [mw.EPISODES, mw.EPISODES[::-1]])
def test_split_episodes_one_draw_per_episode_in_order(seed, episodes):
    """Split: the i-th episode takes the i-th draw rng.choice(3, p=split) of PCG64(seed)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    probs = [0.6, 0.2, 0.2]
    expected = {e: int(rng.choice(3, p=probs)) for e in episodes}
    assert split_episodes(episodes, probs, seed) == expected


@dataclasses.dataclass
class Run:
    code: int
    err: str
    out: Path
    report: dict | None

    def failed(self) -> None:
        """Exit 1 with a one-line message on stderr (Interface)."""
        lines = self.err.strip().splitlines()
        assert self.code == 1 and len(lines) == 1 and "Traceback" not in self.err


def run_main(root: Path, repo: Path, config: dict, relabel=False, remove=None) -> Run:
    """Writes a world under root (without `remove`) and runs main from another directory."""
    path = mw.write(root, repo, mw.frames(relabel), config)
    if remove:
        (root / remove).unlink()
    (root / "cwd").mkdir()
    err, cwd = io.StringIO(), os.getcwd()
    os.chdir(root / "cwd")
    try:
        with contextlib.redirect_stderr(err):
            code = main([str(path)])
    finally:
        os.chdir(cwd)
    report = root / "out/report.json"
    report = json.loads(report.read_text()) if report.exists() else None
    return Run(code, err.getvalue(), root / "out", report)


SCENARIOS = {  # name: (relabel, config changes)
    "base": (False, {}),
    "relabelled": (True, {}),  # labels change, the split does not
    "one_b": (False, {"min_episodes": 7}),  # Q = beta, delta; (b) delta only; no (c)
    "unqualified": (False, {"min_episodes": 100}),  # no (b), no (c)
    "no_test": (False, {"split": ["0.5", "0.5", "0"]}),
}


@pytest.fixture(scope="module", params=list(SCENARIOS))
def world(request, tmp_path_factory, repo):
    """(name, the run of main, the oracle's report)."""
    relabel, changes = SCENARIOS[request.param]
    config = {**mw.CONFIG, **changes}
    run = run_main(tmp_path_factory.mktemp(request.param), repo, config, relabel)
    return request.param, run, mw.expected(mw.frames(relabel), config)


def test_worlds_exercise_the_spec():
    """beta has no (b) template, (c) is built for alpha and delta only (none with one (b)
    template), gamma (no template) is in F^te; counting failed records or only labelled
    episodes would change the split."""
    fs = mw.frames()
    report = mw.expected(fs, mw.CONFIG)
    assert report["qualified"] == ["alpha", "beta", "delta"]
    assert (
        list(report["b"]["pairs"]) == list(report["c"]["pairs"]) == ["alpha", "delta"]
    )
    assert report["a"]["agreement"]["test"]["per_label"]["gamma"]["n_true"] > 0
    one_b = mw.expected(fs, {**mw.CONFIG, **SCENARIOS["one_b"][1]})
    assert list(one_b["b"]["pairs"]) == ["delta"] and one_b["c"]["pairs"] == {}
    records = mw.parts([dataclasses.replace(f, status="ok") for f in fs], mw.CONFIG)
    assert {e: records[e] for e in report["part"]} != report["part"]
    labelled = mw.parts([f for f in mw.frames(True) if f.label], mw.CONFIG)
    assert any(labelled[e] != report["part"][e] for e in labelled)
    assert all(
        mw.expected(mw.frames(r), {**mw.CONFIG, **c}) for r, c in SCENARIOS.values()
    )


def normal(test: dict) -> dict:
    """(7) keys of one set: errors sorted, confusion without zero cells."""
    per_label = {
        k: {**v, "errors": sorted(v["errors"])} for k, v in test["per_label"].items()
    }
    rows = {y: {p: n for p, n in r.items() if n} for y, r in test["confusion"].items()}
    confusion = {y: r for y, r in rows.items() if r}
    return {
        **{k: test[k] for k in mw.CRITERIA},
        "per_label": per_label,
        "confusion": confusion,
    }


def test_report_keys_and_qualified(world):
    _, run, exp = world
    assert run.code == 0, run.err
    assert set(run.report) == {
        "split",
        "qualified",
        "a",
        "b",
        "c",
        "agreement_matrices",
    }
    assert sorted(run.report["qualified"]) == sorted(exp["qualified"])
    for v in "abc":
        assert set(run.report[v]) == {"pairs", *mw.CRITERIA}
        assert set(exp[v]["pairs"]) <= set(run.report[v]["pairs"])


@pytest.mark.parametrize("variant", ["a", "b", "c"])
@pytest.mark.parametrize("crit", mw.CRITERIA)
def test_thresholds_and_test_metrics(world, variant, crit):
    """(6) on F^va (ties: first in grid order), then (7) on F^te and by g_f."""
    _, run, exp = world
    got, want = run.report[variant][crit], exp[variant][crit]
    assert set(got) == {"thresholds", "test"}
    assert got["thresholds"] == want["thresholds"]
    assert set(got["test"]) == {*mw.CRITERIA, "per_label", "confusion", "by_flag"}
    assert normal(got["test"]) == {
        k: v for k, v in want["test"].items() if k != "by_flag"
    }
    assert set(got["test"]["by_flag"]) == {"true", "false"}
    for g in ("true", "false"):
        assert normal(got["test"]["by_flag"][g]) == want["test"]["by_flag"][g]


@pytest.mark.parametrize("crit", mw.CRITERIA)
def test_agreement_matrices(world, crit):
    """(8): per class with a non-empty F_k, |F_k| and the 4 x 4 matrix (VLM, a, b, c)."""
    _, run, exp = world
    got = run.report["agreement_matrices"][crit]
    shown = {k: m for k, m in got.items() if m and m.get("matrix") is not None}
    assert set(got) <= set(mw.CLASSES)
    assert {k: (m["n"], m["matrix"]) for k, m in shown.items()} == exp[
        "agreement_matrices"
    ][crit]
    assert all(len(m["labellers"]) == 4 for m in shown.values())


def test_split_frames_per_label(world):
    """`split`: per part, frames per label of K and other, zeros included (the names of the
    other entries are not fixed by the spec)."""
    _, run, exp = world
    split = run.report["split"]
    split = list(split.values()) if isinstance(split, dict) else split
    labels = set(mw.LABELS)
    found = [
        [v for v in p.values() if isinstance(v, dict) and set(v) == labels]
        for p in split
    ]
    assert found == [[counts] for counts in exp["split"]]


def test_template_images(world):
    """templates/<variant>/<class>[__<member>]_{template,mask}.png: RGB, alpha < 255 black;
    8-bit grey, 255 in the mask."""
    _, run, exp = world
    root = run.out / "templates"
    files = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )
    kinds = ("template", "mask")
    assert files == sorted(f"{s}_{kind}.png" for s in exp["images"] for kind in kinds)
    for stem, (rgb, mask) in exp["images"].items():
        template, grey = (Image.open(root / f"{stem}_{kind}.png") for kind in kinds)
        assert template.mode == "RGB" and np.array_equal(np.asarray(template), rgb)
        assert grey.mode == "L" and np.array_equal(np.asarray(grey) == 255, mask)


def test_report_md_and_progress(world):
    """report.md: the report as tables; progress on stderr, one line per frame, variant, part."""
    name, run, exp = world
    text = (run.out / "report.md").read_text()
    assert "|" in text and all(k in text for k in mw.CLASSES)
    if name == "base":
        assert len(run.err.strip().splitlines()) >= 3 * exp["n_scored"]


def test_patches_per_class_up_to_the_number_of_patches(tmp_path, repo):
    """16 x 8 frames, P = 4: 8 patches; n_P = 8 gives (c) the whole frame."""
    run = run_main(tmp_path, repo, {**mw.CONFIG, "patches_per_class": 8})
    assert run.code == 0, run.err
    for k in ("alpha", "delta"):
        mask = np.asarray(Image.open(run.out / f"templates/c/{k}_mask.png"))
        assert (mask == 255).all()


FAILURES = {  # config changes, removed file
    "patches_per_class above the number of patches": ({"patches_per_class": 9}, None),
    "empty validation set": ({"split": ["0.5", "0", "0.5"]}, None),
    "unreadable labels": ({}, "labels/labels.jsonl"),
}


@pytest.mark.parametrize("changes, remove", FAILURES.values(), ids=FAILURES)
def test_failures_exit_1_without_report(tmp_path, repo, changes, remove):
    """No report, and no progress line: an empty validation set fails before test frames are
    scored."""
    run = run_main(tmp_path, repo, {**mw.CONFIG, **changes}, remove=remove)
    run.failed()
    assert run.report is None


def test_runs_as_module(tmp_path, repo):
    """`python -m mcpref.gui.evaluate <config path>`, here on an empty validation set."""
    path = mw.write(
        tmp_path, repo, mw.frames(), {**mw.CONFIG, "split": ["0.5", "0", "0.5"]}
    )
    argv = [sys.executable, "-m", "mcpref.gui.evaluate", str(path)]
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=8, check=False, cwd=tmp_path
    )
    Run(proc.returncode, proc.stderr, tmp_path / "out", None).failed()
