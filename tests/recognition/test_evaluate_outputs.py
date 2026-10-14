"""The outputs of `main` of specs/gui-masks.md on the worlds of masks_world.py against the
oracle of masks_expected.py: report.json, templates, report.md; features.npz in
test_evaluate_features.py."""

import dataclasses

import masks_expected as me
import masks_world as mw
import numpy as np
import pytest
from masks_run import run_main
from metrics_oracle import normal
from PIL import Image
from progress_bars import Bar, assert_progress, screen

REPORT_KEYS = {"split", "qualified", *mw.VARIANTS, "agreement_matrices"}
TEST_KEYS = {*mw.CRITERIA, "per_label", "confusion", "by_flag"}


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
            assert sorted(map(str, values)) == sorted(map(str, [len(pairs), sizes])), (
                v,
                k,
            )


@pytest.mark.parametrize("variant", list(mw.VARIANTS))
@pytest.mark.parametrize("crit", mw.CRITERIA)
def test_thresholds_and_test_metrics(world, variant, crit):
    """(6) on F^va (ties: first in grid order), then (7) on F^te and by g_f."""
    _, run, exp = world
    got, want = run.report[variant][crit], exp[variant][crit]
    assert set(got) == {"thresholds", "test"}
    assert got["thresholds"] == want["thresholds"]
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


def test_split_frames_per_label(world):
    """`split`: per part, frames per label of K and other, zeros included (the names of the
    other entries of a part are not fixed by the spec)."""
    _, run, exp = world
    split = run.report["split"]
    parts = [p for k, p in split.items() if k != "folds"]
    labels = set(mw.LABELS)
    found = [
        [v for v in p.values() if isinstance(v, dict) and set(v) == labels]
        for p in parts
    ]
    assert found == [[counts] for counts in exp["split"]]


def test_split_folds(world):
    """`split.folds`: in fold order, the number of training episodes and of labelled training
    frames of each fold of (13), empty folds included."""
    _, run, exp = world
    want = [{"episodes": n, "frames": f} for n, f in exp["folds"]]
    assert run.report["split"]["folds"] == want


def test_template_images(world):
    """templates/<variant>/<class>[__<member>]_{template,mask}.png: RGB, alpha < 255 black
    ((f): 8-bit grey floor(T / 1000)); 8-bit grey, 255 in the mask."""
    _, run, exp = world
    root = run.out / "templates"
    files = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )
    kinds = ("template", "mask")
    assert files == sorted(f"{s}_{kind}.png" for s in exp["images"] for kind in kinds)
    for stem, (image, mask) in exp["images"].items():
        template, grey = (Image.open(root / f"{stem}_{kind}.png") for kind in kinds)
        assert template.mode == ("L" if stem.startswith("f/") else "RGB"), stem
        assert np.array_equal(np.asarray(template), image), stem
        assert grey.mode == "L" and np.array_equal(np.asarray(grey) == 255, mask)
        assert set(np.unique(np.asarray(grey))) <= {0, 255}


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
    for k in ("alpha", "delta"):
        mask = np.asarray(Image.open(run.out / f"templates/c/{k}_mask.png"))
        assert (mask == 255).all()


def test_empty_catalogue(tmp_path, repo):
    """K empty, every label `other`: the line `templates catalogue: nothing to do`, Q and (c)
    empty, the frame bars at their totals; no class has a score, so (5) predicts `other`
    everywhere: no template, no column, no matrix, and every grid point ties in (6)."""
    fs = [
        dataclasses.replace(f, label="other") if f.label in mw.CLASSES else f
        for f in mw.frames()
    ]
    tr, va, te = me.labelled(fs, me.parts(fs, mw.CONFIG))
    run = run_main(tmp_path, repo, mw.CONFIG, fs, catalogue={})
    assert run.code == 0, run.err
    idle = ("templates catalogue", "templates b", "templates c")
    drawn = [
        ("features train", len(tr), "frame"),
        ("score validation", 6 * len(va), "frame"),
    ]
    drawn += [("thresholds", 6 * 27, "point"), ("score test", 6 * len(te), "frame")]
    assert screen(run.err) == [f"{d}: nothing to do" for d in idle] + [
        Bar(d, n, n, unit) for d, n, unit in drawn
    ]
    assert_progress(run.err)
    first = {"tau": 12, "theta": 0.9, "mu": 0.0}
    assert run.report["qualified"] == []
    for v in mw.VARIANTS:
        assert run.report[v]["pairs"] == {}
        for crit in mw.CRITERIA:
            assert run.report[v][crit]["thresholds"] == first
            assert run.report[v][crit]["test"]["agreement"] == 1.0
    assert run.report["agreement_matrices"] == {crit: {} for crit in mw.CRITERIA}
    assert not [p for p in (run.out / "templates").rglob("*") if p.is_file()]
    with np.load(run.out / "features.npz", allow_pickle=False) as z:
        assert z["columns"].shape == (0,) and list(z["labels"]) == ["other"]
        assert z["train_x"].shape == (len(tr), 0)
        assert all(
            set(z[f"test_pred_{v}_{c}"]) == {"other"}
            for v in mw.VARIANTS
            for c in mw.CRITERIA
        )
