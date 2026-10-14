"""What `main` of specs/gui-masks.md should produce on a world of masks_world.py, by the naive
references and the pairs of masks_builds.py: report, images, features, bar totals."""

import itertools
import math
from collections import Counter
from fractions import Fraction

import drawn_oracle as d
import masks_builds as b
import masks_oracle as o
import masks_world as mw
import metrics_oracle as mo
import numpy as np


def parts(fs: list, config: dict) -> dict:
    """Split: one draw per episode of the `ok` records, in order of first appearance."""
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    probs = [float(Fraction(p)) for p in config["split"]]
    episodes = dict.fromkeys(f.relpath for f in fs if f.status == "ok")
    return {e: int(rng.choice(3, p=probs)) for e in episodes}


def columns(pairs: dict) -> list:
    """(14): (name, variant, class, suffix): variants, classes, members in order."""
    return [
        (f"{v}/{k}{s}", v, k, s)
        for v in mw.VARIANTS
        for k in mw.CLASSES
        for s in pairs[v].get(k, {})
    ]


def view(v: str, x: np.ndarray, e: np.ndarray) -> tuple:
    """The image (4), (14) compare in V and the bound's scale: E and 1000 for (f)."""
    return (e, 1000) if v == "f" else (x, 1)


def features(f, x: dict, e: dict, cols: list, pairs: dict, own: dict | None, q) -> list:
    """(14) of frame f; `own`: f's fold pairs for the (b), (c) columns (training frames)."""
    out = []
    for _, v, k, s in cols:
        p = (own if own is not None and v in "bc" else pairs)[v].get(k, {}).get(s)
        img, scale = view(v, x[f.id], e[f.id])
        out.append(math.nan if p is None else o.feature(img, *p, q) / scale)
    return out


def evaluation(te: list, yhat: list) -> dict:
    """(7) on F^te and on its parts by g_f."""

    def on(rows: list) -> dict:
        return mo.metrics(*([r[i] for r in rows] for i in range(3)), mw.LABELS)

    rows = [(f.id, f.label, p, f.flag) for f, p in zip(te, yhat)]
    flags = {str(g).lower(): on([r for r in rows if r[3] == g]) for g in (True, False)}
    return {**on(rows), "by_flag": flags}


def variant(v: str, pairs: dict, x: dict, e: dict, va: list, te: list, grid: tuple):
    """(4)-(7) of one variant: criterion -> {thresholds, test}, and the test predictions."""
    by_class = {k: list(p.values()) for k, p in pairs.items()}
    s = {}
    for f, t in itertools.product(va + te, grid[0]):
        img, scale = view(v, x[f.id], e[f.id])
        s[f.id, t] = {k: o.score(img, p, scale * t) for k, p in by_class.items()}
    preds = {
        pt: [o.predict(s[f.id, pt[0]], mw.CLASSES, *pt[1:]) for f in va]
        for pt in itertools.product(*grid)
    }
    out, yhats = {}, {}
    for crit, (t, th, mu) in mo.choose([f.label for f in va], preds, grid).items():
        yhats[crit] = [o.predict(s[f.id, t], mw.CLASSES, th, mu) for f in te]
        thresholds = {"tau": t, "theta": float(th), "mu": float(mu)}
        out[crit] = {"thresholds": thresholds, "test": evaluation(te, yhats[crit])}
    return out, yhats


def totals(tr, va, te, qualified, pairs, folds_built, config) -> dict:
    """Interface, progress bars: each bar's total, 0 where its loop has no item."""
    rows = math.ceil(mw.H / config["chunk_rows"])
    n_k = Counter(k for _, built in folds_built.values() for k in built)
    tb = sum(sum(f.label == k for f in tr) + (1 + n_k[k]) * rows for k in qualified)
    tc = len(pairs["c"])
    tc += sum(len(p["c"]) for p, _ in folds_built.values()) if tc else 0
    grid = math.prod(len(config[g]) for g in ("tau_grid", "theta_grid", "mu_grid"))
    return {
        "templates catalogue": len(mw.CATALOGUE),
        "templates b": tb,
        "templates c": tc,
        "features train": len(tr),
        "score validation": 6 * len(va),
        "thresholds": 6 * grid,
        "score test": 6 * len(te),
    }


def labelled(fs: list, part: dict) -> tuple:
    """F^tr, F^va, F^te: `ok` frames with a label other than invalid, no_answer."""
    bad = (None, "invalid", "no_answer")
    used = [f for f in fs if f.status == "ok" and f.label not in bad]
    return tuple([f for f in used if part[f.relpath] == i] for i in range(3))


def counts(tr: list, va: list, te: list) -> list:
    """`split`: frames per label of K and other, per part."""
    per = [Counter(f.label for f in p) for p in (tr, va, te)]
    return [{k: c[k] for k in mw.LABELS} for c in per]


def fold_counts(tr: list, train_eps: list, fold: dict, k: int) -> list:
    """`split.folds`: (training episodes, labelled training frames) per fold."""
    return [
        (sum(fold[ep] == l for ep in train_eps), sum(fold[f.relpath] == l for f in tr))
        for l in range(k)
    ]


def feature_table(frames: dict, x: dict, e: dict, pairs: dict, built: dict, fold, q):
    """(14): column names, part -> rows; training rows with their fold's (b), (c) pairs."""
    cols = columns(pairs)
    rows = {
        p: [features(f, x, e, cols, pairs, None, q) for f in frames[p]] for p in frames
    }
    rows["train"] = [
        features(f, x, e, cols, pairs, built[fold[f.relpath]][0], q)
        for f in frames["train"]
    ]
    return [c[0] for c in cols], rows


def scored(pairs: dict, x: dict, e: dict, va: list, te: list, config: dict) -> dict:
    """(4)-(8): per variant the thresholds and test metrics, the test predictions, and the
    agreement matrices of VLM, (a)-(f)."""
    grid = (
        config["tau_grid"],
        *[[Fraction(t) for t in config[g]] for g in ("theta_grid", "mu_grid")],
    )
    out, labellers = {}, {crit: [[f.label for f in te]] for crit in mw.CRITERIA}
    for v in mw.VARIANTS:
        out[v], yhats = variant(v, pairs[v], x, e, va, te, grid)
        for crit in mw.CRITERIA:
            labellers[crit].append(yhats[crit])
            out[f"test_pred_{v}_{crit}"] = yhats[crit]
    matrices = {c: mo.matrices(mw.CLASSES, labellers[c]) for c in mw.CRITERIA}
    return {**out, "agreement_matrices": matrices}


def expected(fs: list, config: dict) -> dict | None:
    """The run of `main` by the oracle (None: empty validation set)."""
    part = parts(fs, config)
    tr, va, te = labelled(fs, part)
    if not va:
        return None
    x = {f.id: np.repeat(mw.grey(f.y)[..., None], 3, axis=2) for f in tr + va + te}
    e = {i: d.edges(v) for i, v in x.items()}
    qualified = b.qualified(tr, config)
    sp = b.spread(tr, x, qualified, config)
    pairs = {**b.drawn(), "b": b.b_pairs(sp), "c": b.c_pairs(sp, config)}
    train_eps = [ep for ep, p in part.items() if p == 0]
    fold = o.folds(train_eps, config["folds"], config["fold_seed"])
    built = b.fold_pairs(tr, x, qualified, sp, fold, config)
    frames = {"train": tr, "validation": va, "test": te}
    exp = {"qualified": qualified, "part": part, "pairs": pairs, "frames": frames}
    exp.update(images=b.images(pairs), split=counts(tr, va, te))
    exp["folds"] = fold_counts(tr, train_eps, fold, config["folds"])
    exp["fold_classes"] = {
        l: (list(p["b"]), list(p["c"])) for l, (p, _) in built.items()
    }
    exp["totals"] = totals(tr, va, te, qualified, pairs, built, config)
    q = config["feature_quantile"]
    exp["columns"], exp["x"] = feature_table(frames, x, e, pairs, built, fold, q)
    return {**exp, **scored(pairs, x, e, va, te, config)}
