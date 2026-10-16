"""The features shards of `main` of specs/gui-masks.md (14), (S) on the worlds of
masks_world.py against the oracle of masks_expected.py and masks_features.py; the fold
templates (13) on episodes of distinct pixel values; an empty catalogue."""

import dataclasses
import json

import masks_expected as me
import masks_world as mw
import numpy as np
import pytest
from masks_run import Run, npz, run_main, shard_files
from progress_bars import Bar, assert_progress, screen

PARTS = ("train", "validation", "test")
KEYS = {"train": set(), "validation": {"c", "m"}, "test": {"pred"}}


def shards(run: Run, part: str) -> list:
    return [npz(p) for p in shard_files(run.out / f"features/{part}", "features.npz")]


def test_features_header(world):
    """features/header.json: the columns (variants, classes, members in order) and the
    labels (K in catalogue order, then other)."""
    _, run, exp = world
    header = json.loads((run.out / "features/header.json").read_text())
    assert header == {"columns": exp["columns"], "labels": mw.LABELS}


@pytest.mark.parametrize("part", PARTS)
def test_features_shards_aligned_with_the_index(world, part):
    """One features shard per index shard, empty ones included, holding the part's labelled
    frames of that index shard in position order: ids, y (str), g (bool), x (float64, frames
    x N_C); validation also c, m; test also pred."""
    _, run, exp = world
    got, want = shards(run, part), exp["features"][part]
    assert len(got) == len(want) == len(exp["index"])
    for z, w in zip(got, want):
        assert set(z) == {"ids", "y", "g", "x"} | KEYS[part]
        assert z["ids"].dtype.kind == z["y"].dtype.kind == "U" and z["g"].dtype == bool
        assert z["ids"].tolist() == w["ids"] and z["y"].tolist() == w["y"]
        assert z["g"].tolist() == w["g"]
        assert z["x"].dtype == np.float64 and z["x"].shape == w["x"].shape


@pytest.mark.parametrize("part", PARTS)
def test_features_values(world, part):
    """(14): the inverted_cdf quantile of each pair's deviations over its mask; in `train`
    the (b), (c) columns use the frame's fold pair, NaN without one."""
    _, run, exp = world
    for z, w in zip(shards(run, part), exp["features"][part]):
        assert np.array_equal(z["x"], w["x"], equal_nan=True)


def test_validation_best_pairs(world):
    """(4) at every tau of tau_grid: c and |M| of the best pair, int64, frames x 6 x |K| x
    |tau_grid|, both 0 for a class without a pair."""
    _, run, exp = world
    for z, w in zip(shards(run, "validation"), exp["features"]["validation"]):
        for key in ("c", "m"):
            assert z[key].dtype == np.int64 and z[key].shape == w[key].shape
            assert np.array_equal(z[key], w[key]), key


def test_test_predictions(world):
    """(5) at the chosen thresholds: y_hat per variant (a-f) and threshold choice
    (agreement, macro_recall), str, frames x 6 x 2."""
    _, run, exp = world
    for z, w in zip(shards(run, "test"), exp["features"]["test"]):
        assert z["pred"].dtype.kind == "U" and z["pred"].shape == w["pred"].shape
        assert z["pred"].tolist() == w["pred"].tolist()


def test_features_nan_only_in_cross_fitted_columns(world):
    _, run, exp = world
    for part in PARTS:
        x = np.concatenate([z["x"] for z in shards(run, part)])
        nan = [c for c, n in zip(exp["columns"], np.isnan(x).any(axis=0)) if n]
        assert all(c[0] in "bc" for c in nan) and (part == "train" or not nan), nan


def distinct_frames() -> list:
    """Episodes whose frames are uniform with a value of their own (Y 20 + 10 e), labelled
    alpha (even e) or delta (odd e)."""
    out = []
    for e, s in ((e, s) for s in range(2) for e in range(mw.N_EP)):
        y = np.full((mw.H, mw.W), 20 + 10 * e)
        label = ("alpha", "delta")[e % 2]
        out.append(mw.Frame(f"7.0_u{e}_{s}", mw.EPISODES[e], "ok", False, label, y))
    return out


def test_fold_templates_exclude_their_own_fold(tmp_path, repo):
    """(13): with every pixel in the (b) masks and q_phi = 1, a training frame's (b) and (c)
    features are |its value - its fold template's|, never 0, since a fold-l template is a
    value of the other folds; the template of all of F^tr is a training frame's value (the
    lower median), so a fold template holding the frame's own fold would give 0."""
    fs = distinct_frames()
    config = {**mw.CONFIG, "spread_max": 255, "feature_quantile": 1.0}
    run = run_main(tmp_path, repo, config, fs)
    assert run.code == 0, run.err
    part = me.parts(fs, config)
    columns = json.loads((run.out / "features/header.json").read_text())["columns"]
    x = np.concatenate([z["x"] for z in shards(run, "train")])
    for s, k in enumerate(("alpha", "delta")):
        for v in "bc":
            phi = x[:, columns.index(f"{v}/{k}")]
            assert (phi > 0).all(), (v, k, phi)
        t = npz(run.out / f"templates/b/{s:06d}/pairs.npz")["all/T"]
        values = {
            int(mw.grey(f.y)[0, 0]) for f in fs if f.label == k and part[f.relpath] == 0
        }
        assert len(np.unique(t)) == 1 and int(t[0, 0, 0]) in values


@pytest.fixture(scope="module")
def empty(tmp_path_factory, repo) -> tuple:
    """K empty, every label `other`: (the run of main, F^tr, F^va, F^te); read-only."""
    fs = [
        dataclasses.replace(f, label="other") if f.label in mw.CLASSES else f
        for f in mw.frames()
    ]
    root = tmp_path_factory.mktemp("empty")
    run = run_main(root, repo, mw.CONFIG, fs, catalogue={})
    return run, *me.labelled(fs, me.parts(fs, mw.CONFIG))


def test_empty_catalogue_bars(empty):
    """`templates catalogue` and `templates b` without shards, the (c) shards without a
    class: `nothing to do`; `model` over the folds + 1 (c) shards."""
    run, tr, va, te = empty
    assert run.code == 0, run.err
    idle = [f"templates {d}: nothing to do" for d in ("catalogue", "b", "c")]
    drawn = [("features train", len(tr)), ("score validation", len(va))]
    drawn += [("thresholds", len(va)), ("score test", len(te)), ("report", len(te))]
    assert screen(run.err) == [
        Bar("index", 54, 54, "frame"),
        *idle,
        Bar("model", 4, 4, "shard"),
    ] + [Bar(d, n, n, "frame") for d, n in drawn]
    assert_progress(run.err)


def test_empty_catalogue_report(empty):
    """No class has a score, so (5) predicts `other` everywhere: no pair, no matrix, and
    every grid point ties in (6) (the first one chosen)."""
    run, *_ = empty
    first = {"tau": 12, "theta": 0.9, "mu": 0.0}
    assert run.report["qualified"] == []
    for v in mw.VARIANTS:
        assert run.report[v]["pairs"] == {}
        for crit in mw.CRITERIA:
            assert run.report[v][crit]["thresholds"] == first
            assert run.report[v][crit]["test"]["agreement"] == 1.0
    assert run.report["agreement_matrices"] == {crit: {} for crit in mw.CRITERIA}


def test_empty_catalogue_files(empty):
    """Only the empty (c) shards among the templates; no column; `other` predicted."""
    run, *_ = empty
    templates = sorted(
        p.relative_to(run.out).as_posix() for p in run.out.rglob("templates/**/*.*")
    )
    assert templates == [f"templates/c/{s:06d}/masks.npz" for s in range(4)]
    assert all(npz(run.out / t) == {} for t in templates)
    header = json.loads((run.out / "features/header.json").read_text())
    assert header == {"columns": [], "labels": ["other"]}
    test = shards(run, "test")
    assert all(z["x"].shape == (len(z["ids"]), 0) for z in test)
    assert {p for z in test for p in z["pred"].ravel().tolist()} == {"other"}
