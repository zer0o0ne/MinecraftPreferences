"""What `main` of specs/gui-masks.md should produce on a world of masks_world.py, by the naive
references and the pairs of masks_builds.py: index and template shards, scores, report, the
items of the bars; the features shards in masks_features.py."""

import itertools
import math
from collections import Counter
from fractions import Fraction

import drawn_oracle as d
import masks_builds as b
import masks_features as mf
import masks_oracle as o
import masks_world as mw
import metrics_oracle as mo
import numpy as np
from masks_files import plan_shards

BAD = ("invalid", "no_answer")


def split(fs: list, config: dict) -> tuple:
    """Split: one draw per episode in order of first appearance among the `ok` frames in
    position order; the generator's state after each plan shard."""
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    probs = [float(Fraction(p)) for p in config["split"]]
    part, states = {}, []
    for shard in plan_shards(fs):
        for f in (f for f in shard if f.status == "ok" and f.relpath not in part):
            part[f.relpath] = int(rng.choice(3, p=probs))
        states.append(rng.bit_generator.state)
    return part, states


def parts(fs: list, config: dict) -> dict:
    return split(fs, config)[0]


def index(fs: list, part: dict, states: list) -> list:
    """(I): index shard s = the `ok` frames of plan shard s and the state after it."""
    return [
        {
            "frames": [
                {"id": f.id, "episode": f.relpath, "label": f.label, "g": f.flag}
                | {"part": part[f.relpath]}
                for f in shard
                if f.status == "ok"
            ],
            "rng": state,
        }
        for shard, state in zip(plan_shards(fs), states)
    ]


def evaluation(te: list, yhat: list) -> dict:
    """(7) on F^te and on its parts by g_f."""

    def on(rows: list) -> dict:
        return mo.metrics(*([r[i] for r in rows] for i in range(3)), mw.LABELS)

    rows = [(f.id, f.label, p, f.flag) for f, p in zip(te, yhat)]
    flags = {str(g).lower(): on([r for r in rows if r[3] == g]) for g in (True, False)}
    return {**on(rows), "by_flag": flags}


def best_pairs(v: str, pairs: dict, x: dict, e: dict, fs: list, taus: list) -> dict:
    """(4): (frame id, tau) -> class -> (c, |M|) of its best pair in V."""
    by_class = {k: list(p.values()) for k, p in pairs.items()}
    out = {}
    for f, t in itertools.product(fs, taus):
        img, scale = mf.view(v, x[f.id], e[f.id])
        out[f.id, t] = {k: o.best(img, p, scale * t) for k, p in by_class.items()}
    return out


def variant(v: str, pairs: dict, x: dict, e: dict, va: list, te: list, grid: tuple):
    """(4)-(7) of one variant: criterion -> {thresholds, test}, the test predictions, the
    best pairs of the validation and test frames."""
    cm = best_pairs(v, pairs, x, e, va + te, grid[0])
    s = {key: {k: Fraction(*p) for k, p in c.items()} for key, c in cm.items()}
    preds = {
        pt: [o.predict(s[f.id, pt[0]], mw.CLASSES, *pt[1:]) for f in va]
        for pt in itertools.product(*grid)
    }
    out, yhats = {}, {}
    for crit, (t, th, mu) in mo.choose([f.label for f in va], preds, grid).items():
        yhats[crit] = [o.predict(s[f.id, t], mw.CLASSES, th, mu) for f in te]
        thresholds = {"tau": t, "theta": th, "mu": mu}
        out[crit] = {"thresholds": thresholds, "test": evaluation(te, yhats[crit])}
    return out, yhats, cm


def labelled(fs: list, part: dict) -> tuple:
    """F^tr, F^va, F^te: `ok` frames with a label other than invalid, no_answer."""
    used = [f for f in fs if f.status == "ok" and f.label not in BAD]
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


def scored(pairs: dict, x: dict, e: dict, va: list, te: list, config: dict) -> dict:
    """(4)-(8): per variant the thresholds and test metrics, the test predictions and best
    pairs, and the agreement matrices of VLM, (a)-(f)."""
    grid = (
        config["tau_grid"],
        *[[Fraction(t) for t in config[g]] for g in ("theta_grid", "mu_grid")],
    )
    out, labellers = {"best": {}}, {c: [[f.label for f in te]] for c in mw.CRITERIA}
    for v in mw.VARIANTS:
        out[v], yhats, out["best"][v] = variant(v, pairs[v], x, e, va, te, grid)
        for crit in mw.CRITERIA:
            labellers[crit].append(yhats[crit])
            out[f"test_pred_{v}_{crit}"] = yhats[crit]
    matrices = {c: mo.matrices(mw.CLASSES, labellers[c]) for c in mw.CRITERIA}
    return {**out, "agreement_matrices": matrices}


def items(exp: dict, config: dict) -> dict:
    """Interface, progress bars: desc -> the items of each shard of its set, in run order."""
    rows = math.ceil(mw.H / config["chunk_rows"])
    sets = {"index": [len(s["frames"]) for s in exp["index"]]}
    sets["templates catalogue"] = [1] * len(mw.CLASSES)
    sets["templates b"] = [rows] * len(exp["qualified"])
    sets["templates c"] = [len(c) for c in exp["c_shards"]]
    for desc, part in mf.DESCS.items():
        sets[desc] = [len(s["ids"]) for s in exp["features"][part]]
    return sets


def templates(exp: dict, tr: list, x: dict, part: dict, config: dict) -> None:
    """(1)-(3), (10)-(13) into exp: pairs, folds, template shards, images."""
    qualified = b.qualified(tr, config)
    sp = b.spread(tr, x, qualified, config)
    pairs = {**b.drawn(), "b": b.b_pairs(sp), "c": b.c_pairs(sp, config)}
    train_eps = [ep for ep, p in part.items() if p == 0]
    fold = o.folds(train_eps, config["folds"], config["fold_seed"])
    built = b.fold_pairs(tr, x, qualified, sp, fold, config)
    exp.update(qualified=qualified, pairs=pairs, fold=fold, built=built)
    exp["folds"] = fold_counts(tr, train_eps, fold, config["folds"])
    exp["fold_classes"] = {
        l: (list(p["b"]), list(p["c"])) for l, (p, _, _) in built.items()
    }
    exp["catalogue_shards"] = b.catalogue_shards(pairs)
    exp["b_shards"] = b.b_shards(qualified, sp, built)
    exp["c_shards"] = b.c_shards(pairs, built, config["folds"])
    exp["images"] = b.images(pairs, qualified)


def expected(fs: list, config: dict) -> dict | None:
    """The run of `main` by the oracle (None: empty validation set)."""
    part, states = split(fs, config)
    tr, va, te = labelled(fs, part)
    if not va:
        return None
    x = {f.id: np.repeat(mw.grey(f.y)[..., None], 3, axis=2) for f in tr + va + te}
    e = {i: d.edges(v) for i, v in x.items()}
    frames = {"train": tr, "validation": va, "test": te}
    exp = {"part": part, "frames": frames, "index": index(fs, part, states)}
    exp["split"] = counts(tr, va, te)
    templates(exp, tr, x, part, config)
    exp.update(scored(exp["pairs"], x, e, va, te, config))
    exp["columns"], exp["x"] = mf.feature_table(exp, x, e, config["feature_quantile"])
    exp["features"] = mf.shards(exp, config)
    exp["items"] = items(exp, config)
    return exp
