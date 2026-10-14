"""Config, split, the run of (1)-(8) and `main` of specs/gui-masks.md."""

import argparse
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.frame_set import labelled_parts, qualified, read_frames
from mcpref.gui.label_files import OTHER, frame_size, rgb_view
from mcpref.gui.masks import c_pairs, member_pairs, row_starts, variance_pair
from mcpref.gui.metrics import matrices, thresholds
from mcpref.gui.recognise import frame_scores, predictions
from mcpref.gui.report import (
    Named,
    split_report,
    variant_report,
    write_report,
    write_templates,
)
from mcpref.gui.screens import ScreenClass, load_screens
from mcpref.gui.textures import load_textures_config

PROG = "mcpref.gui.evaluate"
VARIANTS = ("a", "b", "c")


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


def b_built(
    cfg: GuiMasksConfig, train: list[dict], q: list[str], size: tuple[int, int]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(2) over Q: (T, sigma_k, M) of each class; bar `templates b`."""
    if not q:
        print("templates b: nothing to do", file=sys.stderr)
        return {}
    frames = {k: [f for f in train if f["label"] == k] for k in q}
    blocks = len(row_starts(size[1], cfg.chunk_rows))
    total = sum(len(fs) + blocks for fs in frames.values())
    built = {}
    with tqdm(total=total, desc="templates b", unit="step") as bar:
        for k, fs in frames.items():
            x = []
            for f in fs:
                x.append(rgb_view(cfg.frames_dir, f, size))
                bar.update()
            built[k] = variance_pair(
                x, cfg.spread_quantile, cfg.spread_max, cfg.chunk_rows, bar.update
            )
    return built


def variant_pairs(
    cfg: GuiMasksConfig,
    classes: dict[str, ScreenClass],
    train: list[dict],
    q: list[str],
    size: tuple[int, int],
) -> dict[str, Named]:
    """(1)-(3): pairs of each variant, named <class>__<member> in (a), <class> in (b), (c)."""
    built = b_built(cfg, train, q, size)
    b = {k: (t, m) for k, (t, _, m) in built.items() if m.any()}
    spreads = {k: s for k, (_, s, _) in built.items()}
    c = c_pairs(b, spreads, cfg.patch, cfg.patches_per_class, cfg.eps)
    textures = load_textures_config(cfg.screens).out_dir
    a = {
        k: {f"{k}__{m}": p for m, p in member_pairs(cls, textures, *size).items()}
        for k, cls in classes.items()
    }
    return {"a": a} | {
        v: {k: {k: p} for k, p in ps.items()} for v, ps in zip("bc", (b, c))
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
    if not val:
        raise ValueError("no labelled validation frames")
    q = qualified(train, names, cfg.min_episodes)
    named = variant_pairs(cfg, classes, train, q, size)
    write_templates(cfg.out_dir, named)
    pairs = {v: {k: list(c.values()) for k, c in named[v].items()} for v in VARIANTS}
    grid = list(itertools.product(cfg.tau_grid, cfg.theta_grid, cfg.mu_grid))
    s_val = frame_scores(cfg.frames_dir, size, cfg.tau_grid, pairs, val, "validation")
    chosen = thresholds(grid, s_val, [f["label"] for f in val], names)
    s_test = frame_scores(cfg.frames_dir, size, cfg.tau_grid, pairs, test, "test")
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
