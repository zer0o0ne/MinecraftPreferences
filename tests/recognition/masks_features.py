"""The features (14) of specs/gui-masks.md and the content of its features shards (Interface,
Produces) on a world of masks_world.py, by the naive references of masks_oracle.py."""

import itertools
import math

import masks_oracle as o
import masks_world as mw
import numpy as np

DESCS = {"features train": "train", "score validation": "validation"}
DESCS["score test"] = "test"


def view(v: str, x: np.ndarray, e: np.ndarray) -> tuple:
    """The image (4), (14) compare in V and the bound's scale: E and 1000 for (f)."""
    return (e, 1000) if v == "f" else (x, 1)


def columns(pairs: dict) -> list:
    """(14): (name, variant, class, suffix): variants, classes, members in order."""
    return [
        (f"{v}/{k}{s}", v, k, s)
        for v in mw.VARIANTS
        for k in mw.CLASSES
        for s in pairs[v].get(k, {})
    ]


def features(f, x: dict, e: dict, cols: list, pairs: dict, own: dict | None, q) -> list:
    """(14) of frame f; `own`: f's fold pairs for the (b), (c) columns (training frames)."""
    out = []
    for _, v, k, s in cols:
        p = (own if own is not None and v in "bc" else pairs)[v].get(k, {}).get(s)
        img, scale = view(v, x[f.id], e[f.id])
        out.append(math.nan if p is None else o.feature(img, *p, q) / scale)
    return out


def feature_table(exp: dict, x: dict, e: dict, q) -> tuple:
    """(14): column names, part -> rows; training rows with their fold's (b), (c) pairs."""
    pairs, built, fold, frames = exp["pairs"], exp["built"], exp["fold"], exp["frames"]
    cols = columns(pairs)
    rows = {
        p: [features(f, x, e, cols, pairs, None, q) for f in fs]
        for p, fs in frames.items()
    }
    rows["train"] = [
        features(f, x, e, cols, pairs, built[fold[f.relpath]][0], q)
        for f in frames["train"]
    ]
    return [c[0] for c in cols], rows


def best_arrays(exp: dict, fs: list, taus: list) -> tuple:
    """`c`, `m`: (c, |M|) of the best pair of (4), frames x 6 x |K| x |tau_grid|, 0 and 0
    for a class without a pair."""
    shape = (len(fs), len(mw.VARIANTS), len(mw.CLASSES), len(taus))
    c, m = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    for (i, f), (j, v), (l, k), (n, t) in itertools.product(
        enumerate(fs), enumerate(mw.VARIANTS), enumerate(mw.CLASSES), enumerate(taus)
    ):
        c[i, j, l, n], m[i, j, l, n] = exp["best"][v][f.id, t].get(k) or (0, 0)
    return c, m


def preds(exp: dict, fs: list) -> np.ndarray:
    """`pred`: y_hat of (5) per variant and threshold choice, frames x 6 x 2."""
    at = {f.id: i for i, f in enumerate(exp["frames"]["test"])}
    rows = [
        [
            [exp[f"test_pred_{v}_{c}"][at[f.id]] for c in mw.CRITERIA]
            for v in mw.VARIANTS
        ]
        for f in fs
    ]
    return np.array(rows, dtype=str).reshape(len(fs), len(mw.VARIANTS), 2)


def shard(exp: dict, part: str, rows: list, config: dict) -> dict:
    """One features shard: the part's frames of one index shard, in position order."""
    fs = [f for f, _ in rows]
    x = np.array([r for _, r in rows], dtype=np.float64)
    out = {"ids": [f.id for f in fs], "y": [f.label for f in fs]}
    out.update(g=[f.flag for f in fs], x=x.reshape(len(fs), len(exp["columns"])))
    if part == "validation":
        out["c"], out["m"] = best_arrays(exp, fs, config["tau_grid"])
    if part == "test":
        out["pred"] = preds(exp, fs)
    return out


def shards(exp: dict, config: dict) -> dict:
    """part -> its features shards, one per index shard, empty ones included."""
    at = {r["id"]: s for s, sh in enumerate(exp["index"]) for r in sh["frames"]}
    out = {}
    for part, fs in exp["frames"].items():
        rows = list(zip(fs, exp["x"][part]))
        out[part] = [
            shard(exp, part, [(f, r) for f, r in rows if at[f.id] == s], config)
            for s in range(len(exp["index"]))
        ]
    return out
