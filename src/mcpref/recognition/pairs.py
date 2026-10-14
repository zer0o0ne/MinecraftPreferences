"""(1)-(3), (10)-(13) of specs/gui-masks.md: the pairs of every variant on all of F^tr and
per fold; bars `templates catalogue`, `templates b`, `templates c`."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from mcpref.gui.label_files import rgb_view
from mcpref.gui.screens import ScreenClass
from mcpref.recognition.drawn_masks import edge_pairs_from, label_pairs, ring_pairs
from mcpref.recognition.masks import (
    Pair,
    member_pairs,
    patch_masks,
    row_starts,
    variance_pair,
)

if TYPE_CHECKING:
    from mcpref.recognition.evaluate import GuiMasksConfig

VARIANTS = ("a", "b", "c", "d", "e", "f")
FOLDED = ("b", "c")
Named = dict[str, dict[str, Pair]]
Built = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def class_pairs(
    k: str, cls: ScreenClass, textures_dir: Path, size: tuple[int, int]
) -> dict[str, dict[str, Pair]]:
    """(1), (10)-(12) of class k: variant -> <class>__<member> -> pair; (f) from the (a),
    (e) pairs, its templates with a channel axis of size 1 (Interface recognise.py)."""
    a, e = member_pairs(cls, textures_dir, *size), ring_pairs(cls, textures_dir, *size)
    f = {m: (t[..., None], mask) for m, (t, mask) in edge_pairs_from(cls, a, e).items()}
    built = {"a": a, "d": label_pairs(cls, textures_dir, *size), "e": e, "f": f}
    return {v: {f"{k}__{m}": p for m, p in ps.items()} for v, ps in built.items()}


def catalogue_pairs(
    classes: dict[str, ScreenClass], textures_dir: Path, size: tuple[int, int]
) -> dict[str, Named]:
    """(1), (10)-(12) over K: variant -> class -> <class>__<member> -> pair; bar
    `templates catalogue`."""
    named = {v: {} for v in VARIANTS if v not in FOLDED}
    if not classes:
        print("templates catalogue: nothing to do", file=sys.stderr)
        return named
    with tqdm(total=len(classes), desc="templates catalogue", unit="class") as bar:
        for k, cls in classes.items():
            for v, ps in class_pairs(k, cls, textures_dir, size).items():
                named[v][k] = ps
            bar.update()
    return named


def held_out(fs: list[dict], fold_of: dict[str, int], built: list[int]) -> list[int]:
    """(13): the built folds l with a frame of F^tr_(-l) among fs."""
    return [l for l in built if any(fold_of[f["relpath"]] != l for f in fs)]


def b_class(
    cfg: "GuiMasksConfig",
    fs: list[dict],
    folds: list[int],
    fold_of: dict[str, int],
    size: tuple[int, int],
    step: Callable[[], None],
) -> dict[int | None, tuple]:
    """(2) of one class on its training frames (key None) and on F^tr_(-l) of each fold l,
    builds with an empty M left out; frames loaded once; step() per load and row block."""
    x = []
    for f in fs:
        x.append(rgb_view(cfg.frames_dir, f, size))
        step()
    subsets = {None: x} | {
        l: [xi for xi, f in zip(x, fs) if fold_of[f["relpath"]] != l] for l in folds
    }
    q, s_max, rows = cfg.spread_quantile, cfg.spread_max, cfg.chunk_rows
    built = {
        key: variance_pair(xs, q, s_max, rows, step) for key, xs in subsets.items()
    }
    return {key: tsm for key, tsm in built.items() if tsm[2].any()}


def b_builds(
    cfg: "GuiMasksConfig",
    train: list[dict],
    q: list[str],
    fold_of: dict[str, int],
    built: list[int],
    size: tuple[int, int],
) -> dict[int | None, Built]:
    """(2), (13) over Q: (T, sigma_k, M) of the classes with a template, on all of F^tr
    (key None) and per built fold; bar `templates b`."""
    builds = {key: {} for key in [None, *built]}
    if not q:
        print("templates b: nothing to do", file=sys.stderr)
        return builds
    frames = {k: [f for f in train if f["label"] == k] for k in q}
    folds = {k: held_out(fs, fold_of, built) for k, fs in frames.items()}
    blocks = len(row_starts(size[1], cfg.chunk_rows))
    total = sum(len(fs) + (1 + len(folds[k])) * blocks for k, fs in frames.items())
    with tqdm(total=total, desc="templates b", unit="step") as bar:
        for k, fs in frames.items():
            for key, b in b_class(cfg, fs, folds[k], fold_of, size, bar.update).items():
                builds[key][k] = b
    return builds


def b_pairs(b: Built) -> dict[str, Pair]:
    """(2): (T, M) of each class."""
    return {k: (t, m) for k, (t, _, m) in b.items()}


def c_inputs(b: dict[int | None, Built]) -> dict[int | None, Built]:
    """(3), (13): the classes taking part in each build of (c), those with a (b) template
    both from all of F^tr (key None) and from the build's frames, when at least two."""
    if len(b[None]) < 2:
        return {}
    both = {key: {k: v for k, v in bk.items() if k in b[None]} for key, bk in b.items()}
    return {key: bk for key, bk in both.items() if len(bk) >= 2}


def c_pairs(
    cfg: "GuiMasksConfig", b: Built, step: Callable[[], None]
) -> dict[str, Pair]:
    """(3): (T, M) of the classes taking part; T = (b)'s template."""
    templates = {k: t for k, (t, _, _) in b.items()}
    spreads = {k: s for k, (_, s, _) in b.items()}
    n_p = cfg.patches_per_class
    masks = patch_masks(templates, spreads, cfg.patch, n_p, cfg.eps, step)
    return {k: (templates[k], m) for k, m in masks.items()}


def c_builds(cfg: "GuiMasksConfig", builds: dict[int | None, Built]) -> dict:
    """(3), (13): each build of (c); bar `templates c`."""
    if not builds:
        print("templates c: nothing to do", file=sys.stderr)
        return {}
    total = sum(len(b) for b in builds.values())
    with tqdm(total=total, desc="templates c", unit="class") as bar:
        return {key: c_pairs(cfg, b, bar.update) for key, b in builds.items()}


def all_pairs(
    cfg: "GuiMasksConfig",
    classes: dict[str, ScreenClass],
    textures_dir: Path,
    train: list[dict],
    q: list[str],
    fold_of: dict[str, int],
    size: tuple[int, int],
) -> tuple[dict[str, Named], dict[int, dict[str, dict[str, Pair]]]]:
    """(1)-(3), (10)-(13): pairs of each variant on all of F^tr, in column order (14);
    per built fold l, the (b) and (c) pairs of each class (class -> pair)."""
    named = catalogue_pairs(classes, textures_dir, size)
    built = sorted({fold_of[f["relpath"]] for f in train})
    b = b_builds(cfg, train, q, fold_of, built, size)
    c = c_builds(cfg, c_inputs(b))
    named["b"] = {k: {k: p} for k, p in b_pairs(b[None]).items()}
    named["c"] = {k: {k: p} for k, p in c.get(None, {}).items()}
    folds = {l: {"b": b_pairs(b[l]), "c": c.get(l, {})} for l in built}
    return {v: named[v] for v in VARIANTS}, folds
