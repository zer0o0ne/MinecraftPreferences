"""Config, split, (6)-(8) and `main` of specs/gui-masks.md."""

import argparse
import itertools
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.frame_set import labelled_parts, qualified, read_frames
from mcpref.gui.label_files import OTHER, frame_size, rgb_view
from mcpref.gui.masks import Pair, bc_pairs, member_pairs, variance_pair
from mcpref.gui.recognise import predict, scores
from mcpref.gui.report import (
    Named,
    pairs_report,
    split_report,
    write_report,
    write_templates,
)
from mcpref.gui.screens import ScreenClass, load_screens
from mcpref.gui.textures import load_textures_config

PROG = "mcpref.gui.evaluate"
VARIANTS = ("a", "b", "c")
Scores = dict[int, list[dict[str, Fraction]]]


@dataclass(frozen=True)
class GuiMasksConfig:
    """Notation: configs/gui-masks.yaml; paths resolved against the repository root."""

    frames_dir: Path
    labels_dir: Path
    screens: Path
    split: tuple[Fraction, ...]
    seed: int
    min_episodes: int
    spread_quantile: float
    spread_max: int
    patch: int
    patches_per_class: int
    eps: float
    tau_grid: tuple[int, ...]
    theta_grid: tuple[Fraction, ...]
    mu_grid: tuple[Fraction, ...]
    chunk_rows: int
    out_dir: Path


def decimals(values: Sequence[str]) -> tuple[Fraction, ...]:
    """Notation: decimal strings read as exact fractions."""
    if not all(isinstance(v, str) for v in values):
        raise ValueError(f"{values} are not decimal strings")
    return tuple(Fraction(v) for v in values)


def check(cfg: GuiMasksConfig) -> None:
    """Notation: the constraints on config fields."""
    if len(cfg.split) != 3 or sum(cfg.split) != 1:
        raise ValueError(f"split {cfg.split} is not three decimals summing to 1")
    if cfg.patch <= 0 or 40 % cfg.patch:
        raise ValueError(f"patch {cfg.patch} does not divide 40")
    if cfg.patches_per_class < 1:
        raise ValueError(f"patches_per_class {cfg.patches_per_class} is not >= 1")
    if not cfg.eps > 0:
        raise ValueError(f"eps {cfg.eps} is not > 0")
    if not all(type(t) is int for t in cfg.tau_grid):
        raise ValueError(f"tau_grid {cfg.tau_grid} holds non-integers")


def load_config(path: Path) -> GuiMasksConfig:
    """Interface: the YAML file loaded into GuiMasksConfig; violated constraints raise."""
    raw = yaml.safe_load(path.read_text())
    paths = {
        k: REPO / raw[k] for k in ("frames_dir", "labels_dir", "screens", "out_dir")
    }
    exact = {k: decimals(raw[k]) for k in ("split", "theta_grid", "mu_grid")}
    cfg = GuiMasksConfig(**raw | paths | exact | {"tau_grid": tuple(raw["tau_grid"])})
    check(cfg)
    return cfg


def split_episodes(
    relpaths: Sequence[str], probs: Sequence[float], seed: int
) -> dict[str, int]:
    """Notation: one rng.choice(3, p=split) per episode in order of first appearance."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return {e: int(rng.choice(3, p=probs)) for e in dict.fromkeys(relpaths)}


def check_patches(cfg: GuiMasksConfig, size: tuple[int, int]) -> None:
    """Notation: n_P must not exceed the number of P x P patches of a W x H frame."""
    width, height = size
    if cfg.patches_per_class > (width // cfg.patch) * (height // cfg.patch):
        raise ValueError(
            f"patches_per_class {cfg.patches_per_class} exceeds the patches"
        )


def variant_pairs(
    cfg: GuiMasksConfig,
    classes: dict[str, ScreenClass],
    train: list[dict],
    q: list[str],
    size: tuple[int, int],
) -> dict[str, Named]:
    """(1)-(3): pairs of each variant, named <class>__<member> in (a), <class> in (b), (c)."""
    built = {
        k: variance_pair(
            [rgb_view(cfg.frames_dir, f, size) for f in train if f["label"] == k],
            cfg.spread_quantile,
            cfg.spread_max,
            cfg.chunk_rows,
        )
        for k in q
    }
    b, c = bc_pairs(built, cfg.patch, cfg.patches_per_class, cfg.eps)
    textures = load_textures_config(cfg.screens).out_dir
    a = {
        k: {f"{k}__{m}": p for m, p in member_pairs(cls, textures, *size).items()}
        for k, cls in classes.items()
    }
    return {"a": a} | {
        v: {k: {k: p} for k, p in ps.items()} for v, ps in zip("bc", (b, c))
    }


def frame_scores(
    cfg: GuiMasksConfig,
    pairs: dict[str, dict[str, list[Pair]]],
    frames: list[dict],
    size: tuple[int, int],
    part: str,
) -> dict[str, Scores]:
    """(4) per variant and tau of tau_grid, for each frame of `part` in order; progress
    on stderr."""
    out = {v: {tau: [] for tau in cfg.tau_grid} for v in pairs}
    for v, p in pairs.items():
        for i, f in enumerate(frames, 1):
            x = rgb_view(cfg.frames_dir, f, size)
            for tau in cfg.tau_grid:
                out[v][tau].append(scores(x, p, tau))
            progress = f"scoring {part}, variant {v}: {i}/{len(frames)} frames"
            print(f"{PROG}: {progress}", file=sys.stderr)
    return out


def share(num: int, den: int) -> Fraction | None:
    """(6)-(8): an exact ratio; null when the denominator is 0."""
    return Fraction(num, den) if den else None


def agreement(y: list[str], y_hat: list[str]) -> Fraction | None:
    """(6): share of frames with y_hat = y."""
    return share(sum(a == b for a, b in zip(y, y_hat)), len(y))


def macro_recall(y: list[str], y_hat: list[str]) -> Fraction | None:
    """(6): mean over the labels present in y of the share of their frames predicted."""
    hits = Counter(a for a, b in zip(y, y_hat) if a == b)
    recalls = [Fraction(hits[k], n) for k, n in Counter(y).items()]
    return sum(recalls, Fraction(0)) / len(recalls) if recalls else None


CRITERIA = {"agreement": agreement, "macro_recall": macro_recall}


def predictions(s: Scores, point: tuple, order: list[str]) -> list[str]:
    """(5) at the grid point (tau, theta, mu)."""
    tau, theta, mu = point
    return [predict(x, order, theta, mu) for x in s[tau]]


def choose(cfg: GuiMasksConfig, s: Scores, y: list[str], order: list[str]) -> dict:
    """(6): per criterion, the first grid point (tau outermost, mu innermost) maximising it."""
    if not y:
        raise ValueError("no labelled validation frames")
    grid = itertools.product(cfg.tau_grid, cfg.theta_grid, cfg.mu_grid)
    preds = [(point, predictions(s, point, order)) for point in grid]
    return {
        name: max(preds, key=lambda pp: criterion(y, pp[1]))[0]
        for name, criterion in CRITERIA.items()
    }


def label_metrics(y: list[str], y_hat: list[str], ids: list[str], k: str) -> dict:
    """(7): counts, precision, recall, F1 and misclassified ids of label k."""
    tp = sum(a == b == k for a, b in zip(y, y_hat))
    n_true, n_pred = y.count(k), y_hat.count(k)
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "precision": share(tp, n_pred),
        "recall": share(tp, n_true),
        "f1": share(2 * tp, n_true + n_pred),
        "errors": [i for i, a, b in zip(ids, y, y_hat) if a == k != b],
    }


def metrics(frames: list[dict], y_hat: list[str], labels: list[str]) -> dict:
    """(7): agreement, macro recall, per-label metrics, confusion matrix (y x y_hat)."""
    y, ids = [f["label"] for f in frames], [f["id"] for f in frames]
    pairs = Counter(zip(y, y_hat))
    return {
        "agreement": agreement(y, y_hat),
        "macro_recall": macro_recall(y, y_hat),
        "per_label": {k: label_metrics(y, y_hat, ids, k) for k in labels},
        "confusion": {a: {b: pairs[a, b] for b in labels} for a in labels},
    }


def evaluation(frames: list[dict], y_hat: list[str], labels: list[str]) -> dict:
    """(7): metrics on F^te, and on its frames with g_f true / false."""
    by_flag = {}
    for name, flag in (("true", True), ("false", False)):
        keep = [i for i, f in enumerate(frames) if f["line"]["isGuiInventory"] == flag]
        by_flag[name] = metrics(
            [frames[i] for i in keep], [y_hat[i] for i in keep], labels
        )
    return metrics(frames, y_hat, labels) | {"by_flag": by_flag}


def agreement_matrix(labellers: dict[str, list[str]], k: str) -> dict:
    """(8): A_k[i, j] over F_k, the frames to which some labeller assigns k."""
    f_k = [row for row in zip(*labellers.values()) if k in row]
    n = range(len(labellers))
    matrix = [[share(sum(r[i] == r[j] for r in f_k), len(f_k)) for j in n] for i in n]
    return {
        "n": len(f_k),
        "labellers": list(labellers),
        "matrix": matrix if f_k else None,
    }


def matrices(test: list[dict], preds: dict, names: list[str]) -> dict:
    """(8) per threshold choice: labellers VLM, (a), (b), (c); one matrix per class."""
    y = [f["label"] for f in test]
    return {
        c: {
            k: agreement_matrix({"vlm": y} | {v: preds[v][c] for v in VARIANTS}, k)
            for k in names
        }
        for c in CRITERIA
    }


def variant_report(
    named: Named, chosen: dict, preds: dict, test: list[dict], names: list[str]
) -> dict:
    """Interface report.json, one variant: pairs; thresholds and (7) per choice."""
    return {"pairs": pairs_report(named, names)} | {
        c: {
            "thresholds": dict(zip(("tau", "theta", "mu"), point)),
            "test": evaluation(test, preds[c], names + [OTHER]),
        }
        for c, point in chosen.items()
    }


def run(cfg: GuiMasksConfig) -> dict:
    """Notation, (1)-(8): templates written to out_dir; returns the report."""
    classes = load_screens(cfg.screens)
    names = list(classes)
    size = frame_size(cfg.frames_dir)
    check_patches(cfg, size)
    episodes, frames = read_frames(cfg.frames_dir, cfg.labels_dir)
    parts = split_episodes(episodes, [float(p) for p in cfg.split], cfg.seed)
    train, val, test = labelled_parts(frames, parts)
    q = qualified(train, names, cfg.min_episodes)
    named = variant_pairs(cfg, classes, train, q, size)
    write_templates(cfg.out_dir, named)
    pairs = {v: {k: list(c.values()) for k, c in named[v].items()} for v in VARIANTS}
    s_val = frame_scores(cfg, pairs, val, size, "validation")
    y_val = [f["label"] for f in val]
    chosen = {v: choose(cfg, s_val[v], y_val, names) for v in VARIANTS}
    s_test = frame_scores(cfg, pairs, test, size, "test")
    preds = {
        v: {c: predictions(s_test[v], point, names) for c, point in chosen[v].items()}
        for v in VARIANTS
    }
    split = split_report(parts, [train, val, test], names + [OTHER])
    report = {"split": split, "qualified": q}
    report |= {
        v: variant_report(named[v], chosen[v], preds[v], test, names) for v in VARIANTS
    }
    return report | {"agreement_matrices": matrices(test, preds, names)}


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.gui.evaluate <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        write_report(cfg.out_dir, run(cfg))
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
