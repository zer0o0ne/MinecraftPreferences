"""(S) of specs/gui-masks.md: (4), (5), (14) and the loops over frames, frame-outer, with the
model in memory; bars `features train`, `score validation`, `thresholds`, `score test`."""

import itertools
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from mcpref import REPO
from mcpref.gui.label_files import OTHER, rgb_view
from mcpref.recognition.drawn_masks import SCALE, edges
from mcpref.recognition.frame_set import State, part_frames
from mcpref.recognition.masks import Pair, distance
from mcpref.recognition.metrics import CRITERIA, choose
from mcpref.recognition.pairs import VARIANTS
from mcpref.recognition.stored import (
    ALL,
    FEATURES,
    PARTS,
    THRESHOLDS,
    Model,
    read_part,
    read_thresholds,
    shard_bar,
    write_features,
    write_thresholds,
)
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards

if TYPE_CHECKING:
    from mcpref.recognition.evaluate import GuiMasksConfig

Pairs = Mapping[str, Sequence[Pair]]
BARS = ("features train", "score validation", "score test")


def view(v: str, x: np.ndarray) -> np.ndarray:
    """Interface: the image variant v compares, E(x_f) with a channel axis of size 1 for
    (f), x_f otherwise."""
    return edges(x)[..., None] if v == "f" else x


def unit(v: str) -> int:
    """Interface: d^f is held in units of 1/1000: the bound is 1000 tau in (4) and the
    quantile is divided by 1000 in (14)."""
    return SCALE if v == "f" else 1


def deviations(x: np.ndarray, pairs: Pairs) -> dict[str, list[np.ndarray]]:
    """(4), (14): d_T(f, p) over p in M, per pair of each class."""
    return {k: [distance(x[m], t[m]) for t, m in ps] for k, ps in pairs.items()}


def best(
    d: Mapping[str, Sequence[np.ndarray]], bound: int
) -> dict[str, tuple[int, int]]:
    """(4): (c, |M|) of the best pair of each class, the first of the largest c / |M|,
    c = |{p in M : d <= bound}|; classes without a pair absent."""
    cms = {
        k: [(int(np.count_nonzero(di <= bound)), di.size) for di in ds]
        for k, ds in d.items()
    }
    return {k: max(cm, key=lambda c_m: Fraction(*c_m)) for k, cm in cms.items() if cm}


def scores(x: np.ndarray, pairs: Pairs, tau: int) -> dict[str, tuple[int, int]]:
    """(4): class -> (c, |M|) of its best pair, classes without a pair absent."""
    return best(deviations(x, pairs), tau)


def exact(s: Mapping[str, tuple[int, int]]) -> dict[str, Fraction]:
    """(4): S_k = c / |M| as an exact fraction."""
    return {k: Fraction(*c_m) for k, c_m in s.items()}


def ranked(scores: Mapping[str, Fraction], order: Sequence[str]) -> tuple | None:
    """(5): k*, S_k* and S_(2) (0 if only one score); ties in `order`; None: no score."""
    present = [k for k in order if k in scores]
    if not present:
        return None
    top = max(present, key=scores.__getitem__)
    second = max((scores[k] for k in present if k != top), default=Fraction(0))
    return top, scores[top], second


def decide(top: tuple | None, theta: Fraction, mu: Fraction) -> str:
    """(5): k* if S_k* >= theta and S_k* - S_(2) >= mu, else `other`."""
    if top is None:
        return OTHER
    k, s, second = top
    return k if s >= theta and s - second >= mu else OTHER


def predict(
    scores: Mapping[str, Fraction], order: Sequence[str], theta: Fraction, mu: Fraction
) -> str:
    """(5): the predicted label of a frame."""
    return decide(ranked(scores, order), theta, mu)


def quantile(d: np.ndarray, q: float) -> float:
    """(14): the inverted_cdf quantile at q of the |M| values d, as float64."""
    return float(np.quantile(d, q, method="inverted_cdf"))


def pair_feature(x: np.ndarray, t: np.ndarray, m: np.ndarray, q: float) -> float:
    """(14) for one pair: the quantile at q of d_T(f, p), p in M."""
    return quantile(distance(x[m], t[m]), q)


def build_pairs(model: Model, b: str) -> dict[str, Pairs]:
    """(4), (14): per variant, class -> its pairs for the frames scored with build b:
    (b), (c) from b (none for a class without), the others built on all of F^tr."""
    by_v = model[ALL] | model[b]
    return {
        v: {k: list(cols.values()) for k, cols in by_v[v].items()} for v in VARIANTS
    }


def frame_deviations(x: np.ndarray, pairs: Mapping[str, Pairs]) -> dict:
    """(4), (14): per variant, d^V_T of each pair of each class on frame x_f."""
    return {v: deviations(view(v, x), pairs[v]) for v in VARIANTS}


def feature_row(d: Mapping, own: Mapping[str, Pairs], q: float) -> list[float]:
    """(14): phi_f in column order (the pairs `own` built on all of F^tr); NaN for the
    column of a class without its pair in d."""
    row = []
    for v in VARIANTS:
        for k, ps in own[v].items():
            ds = d[v].get(k, [])
            row += (
                [quantile(di, q) / unit(v) for di in ds] if ds else [np.nan] * len(ps)
            )
    return row


def tau_counts(d: Mapping, names: list[str], tau_grid: Sequence[int]) -> np.ndarray:
    """(4) at every tau: c and |M| of the best pair, 2 x 6 x |K| x |tau_grid|, 0 without."""
    cm = np.zeros((2, len(VARIANTS), len(names), len(tau_grid)), np.int64)
    for i, v in enumerate(VARIANTS):
        for t, tau in enumerate(tau_grid):
            for k, c_m in best(d[v], tau * unit(v)).items():
                cm[:, i, names.index(k), t] = c_m
    return cm


def frame_preds(d: Mapping, chosen: Mapping, names: list[str]) -> list[list[str]]:
    """(4), (5) at each variant's chosen point of each threshold choice: 6 x 2."""
    return [
        [
            predict(exact(best(d[v], tau * unit(v))), names, theta, mu)
            for tau, theta, mu in (chosen[v][c] for c in CRITERIA)
        ]
        for v in VARIANTS
    ]


def train_row(
    x: np.ndarray, f: dict, pairs: Mapping, fold_of: Mapping, q: float
) -> dict:
    """(14) for a frame of F^tr_(l), l = the fold of its episode."""
    d = frame_deviations(x, pairs[f"fold{fold_of[f['episode']]}"])
    return {"x": feature_row(d, pairs[ALL], q)}


def validation_row(
    x: np.ndarray, f: dict, own: Mapping, cfg: "GuiMasksConfig", names: list[str]
) -> dict:
    """(4) at every tau of tau_grid, and (14), for a frame of F^va."""
    d = frame_deviations(x, own)
    cm = tau_counts(d, names, cfg.tau_grid)
    return {"x": feature_row(d, own, cfg.feature_quantile), "c": cm[0], "m": cm[1]}


def test_row(
    x: np.ndarray, f: dict, own: Mapping, q: float, names: list[str], chosen: Mapping
) -> dict:
    """(4), (5) at the chosen points, and (14), for a frame of F^te."""
    d = frame_deviations(x, own)
    return {"x": feature_row(d, own, q), "pred": frame_preds(d, chosen, names)}


def score_part(
    cfg: "GuiMasksConfig", state: State, part: int, row: Callable, shapes: Mapping
) -> None:
    """(S): the features shards of a part, one per index shard, each frame read once;
    bar of the part."""
    out = REPO / cfg.out_dir
    directory, n = out / FEATURES / PARTS[part], [c[part] for c in state.counts]
    h = complete_shards(directory)
    with shard_bar(BARS[part], len(n), h, sum(n), sum(n[:h]), "frame") as update:
        for s in range(h, len(n)):
            frames, rows = part_frames(out, s, part), []
            for f in frames:
                rows.append(row(rgb_view(REPO / cfg.frames_dir, f, state.size), f))
                update()
            write_features(directory, s, frames, rows, shapes)


def tops(c: np.ndarray, m: np.ndarray, names: list[str]) -> list[tuple | None]:
    """(5) at every tau: k*, S_k*, S_(2) from the stored c, |M| (|K| x |tau_grid|)."""
    by_tau = (zip(names, c[:, t].tolist(), m[:, t].tolist()) for t in range(c.shape[1]))
    return [ranked(exact({k: cm for k, *cm in s if cm[1]}), names) for s in by_tau]


def tally_grid(
    acc: list[list[Counter]], z: Mapping, cfg: "GuiMasksConfig", names: list[str]
) -> None:
    """(6): adds one validation shard: per variant and grid point (tau outermost, mu
    innermost), the frames per (y, y_hat), y_hat by (5) from the stored c, |M|."""
    grid = list(
        itertools.product(range(len(cfg.tau_grid)), cfg.theta_grid, cfg.mu_grid)
    )
    for y, c, m in zip(z["y"].tolist(), z["c"], z["m"]):
        for acc_v, c_v, m_v in zip(acc, c, m):
            top = tops(c_v, m_v, names)
            for p, (t, theta, mu) in enumerate(grid):
                acc_v[p][y, decide(top[t], theta, mu)] += 1


def thresholds(cfg: "GuiMasksConfig", state: State, raw: Mapping) -> dict:
    """(6) from the validation shards read one at a time, written to thresholds.json and
    read from it; bar `thresholds`."""
    path = REPO / cfg.out_dir / THRESHOLDS
    if path.exists():
        print("thresholds: already done", file=sys.stderr)
        return read_thresholds(path)
    points = list(itertools.product(cfg.tau_grid, raw["theta_grid"], raw["mu_grid"]))
    acc = [[Counter() for _ in points] for _ in VARIANTS]
    tally = partial(tally_grid, acc, cfg=cfg, names=state.names)
    read_part(REPO / cfg.out_dir, 1, state.counts, "thresholds", tally)
    chosen = [{c: points[p] for c, p in choose(a).items()} for a in acc]
    write_thresholds(path, dict(zip(VARIANTS, chosen)))
    return read_thresholds(path)


def score(cfg: "GuiMasksConfig", state: State, model: Model, raw: Mapping) -> dict:
    """(S): features/header.json, then the parts train, validation (and (6)), test;
    returns the chosen points of (6)."""
    out, names, q = REPO / cfg.out_dir, state.names, cfg.feature_quantile
    pairs = {b: build_pairs(model, b) for b in model}
    own = pairs[ALL]
    columns = [
        f"{v}/{c}" for v in VARIANTS for cs in model[ALL][v].values() for c in cs
    ]
    header = {"columns": columns, "labels": [*names, OTHER]}
    write_json(out / FEATURES / "header.json", header)
    x = {"x": ((len(columns),), np.float64)}
    row = partial(train_row, pairs=pairs, fold_of=state.fold_of, q=q)
    score_part(cfg, state, 0, row, x)
    cm = ((len(VARIANTS), len(names), len(cfg.tau_grid)), np.int64)
    row = partial(validation_row, own=own, cfg=cfg, names=names)
    score_part(cfg, state, 1, row, x | {"c": cm, "m": cm})
    chosen = thresholds(cfg, state, raw)
    row = partial(test_row, own=own, q=q, names=names, chosen=chosen)
    score_part(cfg, state, 2, row, x | {"pred": ((len(VARIANTS), len(CRITERIA)), str)})
    return chosen
