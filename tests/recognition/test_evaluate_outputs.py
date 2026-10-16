"""The outputs of `main` of specs/gui-masks.md on the worlds of masks_world.py against the
oracle of masks_expected.py: the files of {out_dir}, the template shards, thresholds.json,
report.json, report.md; the features shards in test_evaluate_features.py."""

import json
from fractions import Fraction

import masks_world as mw
import numpy as np
import pytest
from masks_run import config_of, npz, run_main
from metrics_oracle import normal
from PIL import Image

REPORT_KEYS = {"split", "qualified", *mw.VARIANTS, "agreement_matrices"}
TEST_KEYS = {*mw.CRITERIA, "per_label", "confusion", "by_flag"}
PARTS = ("train", "validation", "test")
TEMPLATE_SETS = {"catalogue": "pairs.npz", "b": "pairs.npz", "c": "masks.npz"}


def files(exp: dict) -> list:
    """Produces: every file of {out_dir} after a run that exits 0."""
    n_s = len(exp["index"])
    out = ["run.json", "thresholds.json", "report.json", "report.md"]
    out += [f"index/{s:06d}/index.json" for s in range(n_s)]
    out += [f"features/{p}/{s:06d}/features.npz" for p in PARTS for s in range(n_s)]
    out += ["features/header.json"]
    for kind, name in TEMPLATE_SETS.items():
        n = len(exp[f"{kind}_shards"])
        out += [f"templates/{kind}/{s:06d}/{name}" for s in range(n)]
    out += [
        f"templates/{s}_{k}.png" for s in exp["images"] for k in ("template", "mask")
    ]
    return sorted(out)


def test_output_files(world):
    """Every file of Produces and nothing else: no features.npz, no staged shard."""
    _, run, exp = world
    found = sorted(
        p.relative_to(run.out).as_posix() for p in run.out.rglob("*") if p.is_file()
    )
    assert found == files(exp)


@pytest.mark.parametrize("kind", list(TEMPLATE_SETS))
def test_template_shards(world, kind):
    """templates/catalogue (class s of K: <variant>/<member>/T, M of (a), (d)-(f)),
    templates/b (class s of Q: <build>/T, sigma, M of every build with a non-empty M),
    templates/c (build s of all, fold0, ...: <class> -> its mask), empty .npz included;
    T uint8 H x W x 3 (int64 H x W for (f)), sigma int16, M bool."""
    _, run, exp = world
    for s, want in enumerate(exp[f"{kind}_shards"]):
        got = npz(run.out / f"templates/{kind}/{s:06d}/{TEMPLATE_SETS[kind]}")
        assert sorted(got) == sorted(want), (kind, s)
        for key, a in want.items():
            assert got[key].dtype == dtype(kind, key) and got[key].shape == a.shape
            assert np.array_equal(got[key], a), (kind, s, key)


def dtype(kind: str, key: str):
    """Produces: T uint8 (int64 for (f)), sigma int16, M and the (c) masks bool."""
    if kind == "c" or key.endswith("/M"):
        return np.dtype(bool)
    if key.endswith("/sigma"):
        return np.dtype(np.int16)
    return np.dtype(np.int64 if key.startswith("f/") else np.uint8)


def test_template_images(world):
    """Next to each .npz, the pairs built on all of F^tr: <variant>__<member or
    class>_template.png (RGB, alpha < 255 black; (f): 8-bit grey floor(T / 1000)) and
    _mask.png (8-bit grey, 255 in the mask)."""
    _, run, exp = world
    for stem, (image, mask) in exp["images"].items():
        template, grey = (
            Image.open(run.out / f"templates/{stem}_{kind}.png")
            for kind in ("template", "mask")
        )
        assert template.mode == ("L" if "/f__" in stem else "RGB"), stem
        assert np.array_equal(np.asarray(template), image), stem
        assert grey.mode == "L" and np.array_equal(np.asarray(grey) == 255, mask)
        assert set(np.unique(np.asarray(grey))) <= {0, 255}


def test_thresholds_json(world):
    """(6): per variant, per threshold choice, {tau, theta, mu}, theta and mu as the strings
    of the grids as written in the config."""
    name, run, exp = world
    config = config_of(name)
    strings = {
        g: {Fraction(t): t for t in config[g]} for g in ("theta_grid", "mu_grid")
    }
    want = {
        v: {
            c: {"tau": t["tau"], "theta": strings["theta_grid"][t["theta"]]}
            | {"mu": strings["mu_grid"][t["mu"]]}
            for c, t in ((c, exp[v][c]["thresholds"]) for c in mw.CRITERIA)
        }
        for v in mw.VARIANTS
    }
    assert json.loads((run.out / "thresholds.json").read_text()) == want, name


def test_report_keys_and_qualified(world):
    _, run, exp = world
    assert run.code == 0, run.err
    assert set(run.report) == REPORT_KEYS
    assert sorted(run.report["qualified"]) == sorted(exp["qualified"])
    for v in mw.VARIANTS:
        assert set(run.report[v]) == {"pairs", *mw.CRITERIA}


def test_pairs(world):
    """`pairs`, per class: the number of pairs and their mask sizes (key names not fixed); a
    class without a pair, if listed, with 0 pairs and no size."""
    _, run, exp = world
    for v in mw.VARIANTS:
        got = run.report[v]["pairs"]
        assert set(exp["pairs"][v]) <= set(got) <= set(mw.CLASSES)
        for k in got:
            pairs = exp["pairs"][v].get(k, {})
            sizes = sorted(int(m.sum()) for _, m in pairs.values())
            values = [sorted(x) if isinstance(x, list) else x for x in got[k].values()]
            assert sorted(map(str, values)) == sorted(map(str, [len(pairs), sizes]))


@pytest.mark.parametrize("variant", list(mw.VARIANTS))
@pytest.mark.parametrize("crit", mw.CRITERIA)
def test_thresholds_and_test_metrics(world, variant, crit):
    """(6) on F^va (ties: first in grid order), then (7) on F^te and by g_f."""
    _, run, exp = world
    got, want = run.report[variant][crit], exp[variant][crit]
    assert set(got) == {"thresholds", "test"}
    t = want["thresholds"]
    assert got["thresholds"] == {k: float(v) if k != "tau" else v for k, v in t.items()}
    assert set(got["test"]) == TEST_KEYS
    assert normal(got["test"]) == {
        k: v for k, v in want["test"].items() if k != "by_flag"
    }
    assert set(got["test"]["by_flag"]) == {"true", "false"}
    for g in ("true", "false"):
        assert normal(got["test"]["by_flag"][g]) == want["test"]["by_flag"][g]


@pytest.mark.parametrize("crit", mw.CRITERIA)
def test_agreement_matrices(world, crit):
    """(8): per class with a non-empty F_k, |F_k| and the 7 x 7 matrix (VLM, (a)-(f))."""
    _, run, exp = world
    got = run.report["agreement_matrices"][crit]
    shown = {k: m for k, m in got.items() if m and m.get("matrix") is not None}
    assert set(got) <= set(mw.CLASSES)
    want = exp["agreement_matrices"][crit]
    assert {k: (m["n"], m["matrix"]) for k, m in shown.items()} == want
    assert all(len(m["labellers"]) == 7 for m in shown.values())


def test_split_parts(world):
    """`split`, per part in the order train, validation, test (their names not fixed by the
    spec): the episodes of the part (with or without labelled frames, as in `folds`), its
    labelled frames, and its frames per label of K and other, zeros included."""
    _, run, exp = world
    parts = [p for k, p in run.report["split"].items() if k != "folds"]
    assert [p["episodes"] for p in parts] == [
        sum(d == i for d in exp["part"].values()) for i in range(3)
    ]
    assert [p["per_label"] for p in parts] == exp["split"]
    assert [p["frames"] for p in parts] == [sum(c.values()) for c in exp["split"]]


def test_split_folds(world):
    """`split.folds`: in fold order, the number of training episodes and of labelled training
    frames of each fold of (13), empty folds included."""
    _, run, exp = world
    want = [{"episodes": n, "frames": f} for n, f in exp["folds"]]
    assert run.report["split"]["folds"] == want


def test_report_md(world):
    """report.md: the report as tables, titled `GUI masks report`."""
    _, run, _ = world
    text = (run.out / "report.md").read_text()
    assert text.splitlines()[0] == "# GUI masks report"
    assert "|" in text and all(k in text for k in mw.CLASSES)


def test_patches_per_class_up_to_the_number_of_patches(tmp_path, repo):
    """24 x 20 frames, P = 4: 30 patches; n_P = 30 gives (c) the whole frame."""
    config = {**mw.CONFIG, "patches_per_class": 30}
    run = run_main(tmp_path, repo, config, mw.frames())
    assert run.code == 0, run.err
    masks = npz(run.out / "templates/c/000000/masks.npz")
    assert sorted(masks) == ["alpha", "delta"] and all(m.all() for m in masks.values())
