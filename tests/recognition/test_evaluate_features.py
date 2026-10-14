"""features.npz of `main` of specs/gui-masks.md (14) on the worlds of masks_world.py against
the oracle of masks_expected.py, and the fold templates (13) on episodes of distinct pixel
values."""

import masks_expected as me
import masks_world as mw
import numpy as np
import pytest
from masks_run import run_main
from PIL import Image

REPORT_KEYS = {"split", "qualified", *mw.VARIANTS, "agreement_matrices"}
TEST_KEYS = {*mw.CRITERIA, "per_label", "confusion", "by_flag"}


PARTS = ("train", "validation", "test")
PREDS = [f"test_pred_{v}_{c}" for v in mw.VARIANTS for c in mw.CRITERIA]
FEATURE_KEYS = {"columns", "labels", *PREDS}
FEATURE_KEYS |= {f"{p}_{k}" for p in PARTS for k in ("ids", "y", "g", "x")}


def features(run) -> dict:
    """features.npz, loaded without pickle (no object arrays)."""
    with np.load(run.out / "features.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_features_keys_and_frames(world):
    """features.npz: columns (variants, classes, members in order), labels (K, other), per part
    the frames of the part in frames.jsonl order: ids, y, g; the twelve test predictions."""
    _, run, exp = world
    z = features(run)
    assert set(z) == FEATURE_KEYS
    assert z["columns"].dtype.kind == "U" and z["columns"].tolist() == exp["columns"]
    assert z["labels"].dtype.kind == "U" and z["labels"].tolist() == mw.LABELS
    for p in PARTS:
        fs = exp["frames"][p]
        assert z[f"{p}_ids"].dtype.kind == z[f"{p}_y"].dtype.kind == "U"
        assert z[f"{p}_ids"].tolist() == [f.id for f in fs]
        assert z[f"{p}_y"].tolist() == [f.label for f in fs]
        assert z[f"{p}_g"].dtype == bool and z[f"{p}_g"].tolist() == [
            f.flag for f in fs
        ]
    for key in PREDS:
        assert z[key].dtype.kind == "U" and z[key].tolist() == exp[key], key


@pytest.mark.parametrize("part", PARTS)
def test_features_values(world, part):
    """(14): the inverted_cdf quantile of each pair's deviations over its mask, float64,
    frames x N_C; in `train` the (b), (c) columns use the frame's fold pair, NaN without one."""
    _, run, exp = world
    got = features(run)[f"{part}_x"]
    shape = (len(exp["frames"][part]), len(exp["columns"]))
    want = np.array(exp["x"][part], dtype=np.float64).reshape(shape)
    assert got.dtype == np.float64 and got.shape == want.shape
    assert np.array_equal(got, want, equal_nan=True)


def test_features_nan_only_in_cross_fitted_columns(world):
    _, run, _ = world
    z = features(run)
    nan = [c for c, n in zip(z["columns"], np.isnan(z["train_x"]).any(axis=0)) if n]
    assert all(c[0] in "bc" for c in nan), nan
    assert not np.isnan(z["validation_x"]).any() and not np.isnan(z["test_x"]).any()


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
    z, part = features(run), me.parts(fs, config)
    columns = z["columns"].tolist()
    for k in ("alpha", "delta"):
        for v in "bc":
            phi = z["train_x"][:, columns.index(f"{v}/{k}")]
            assert (phi > 0).all(), (v, k, phi)
        t = np.asarray(Image.open(run.out / f"templates/b/{k}_template.png"))
        values = {
            int(mw.grey(f.y)[0, 0]) for f in fs if f.label == k and part[f.relpath] == 0
        }
        assert len(np.unique(t)) == 1 and int(t[0, 0, 0]) in values
