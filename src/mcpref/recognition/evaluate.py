"""Config, split, the run of (1)-(14) and `main` of specs/gui-masks.md."""

import argparse
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.label_files import OTHER, frame_size
from mcpref.gui.screens import load_screens
from mcpref.gui.textures import load_textures_config
from mcpref.recognition.frame_set import folds, labelled_parts, qualified, read_frames
from mcpref.recognition.metrics import matrices, thresholds
from mcpref.recognition.pairs import VARIANTS, all_pairs
from mcpref.recognition.recognise import part_scores, predictions, train_features
from mcpref.recognition.report import (
    split_report,
    variant_report,
    write_features,
    write_report,
    write_templates,
)

PROG = "mcpref.recognition.evaluate"
TITLE = "GUI masks report"


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
    folds: int
    fold_seed: int
    feature_quantile: float
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
    if type(cfg.min_episodes) is not int or cfg.min_episodes < 1:
        raise ValueError(f"min_episodes {cfg.min_episodes} is not an integer >= 1")
    if cfg.patch <= 0 or 40 % cfg.patch:
        raise ValueError(f"patch {cfg.patch} does not divide 40")
    if cfg.patches_per_class < 1:
        raise ValueError(f"patches_per_class {cfg.patches_per_class} is not >= 1")
    if not cfg.eps > 0:
        raise ValueError(f"eps {cfg.eps} is not > 0")
    if not all(type(t) is int for t in cfg.tau_grid):
        raise ValueError(f"tau_grid {cfg.tau_grid} holds non-integers")
    if type(cfg.folds) is not int or cfg.folds < 2:
        raise ValueError(f"folds {cfg.folds} is not an integer >= 2")
    if type(cfg.fold_seed) is not int:
        raise ValueError(f"fold_seed {cfg.fold_seed} is not an integer")
    if not 0 < cfg.feature_quantile <= 1:
        raise ValueError(f"feature_quantile {cfg.feature_quantile} is not in (0, 1]")


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


def split_frames(cfg: GuiMasksConfig) -> tuple[dict[str, int], list[list[dict]]]:
    """Notation: the split of the episodes and F^tr, F^va, F^te; an empty F^va raises."""
    episodes, frames = read_frames(cfg.frames_dir, cfg.labels_dir)
    parts = split_episodes(episodes, [float(p) for p in cfg.split], cfg.seed)
    labelled = labelled_parts(frames, parts)
    if not labelled[1]:
        raise ValueError("no labelled validation frames")
    return parts, labelled


def recognised(
    cfg: GuiMasksConfig,
    size: tuple[int, int],
    named: dict,
    val: list[dict],
    test: list[dict],
    names: list[str],
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """(4)-(6), (14): F^va scored, thresholds chosen, then F^te scored and predicted;
    returns the choices, the test predictions and the features of F^va, F^te."""
    grid = list(itertools.product(cfg.tau_grid, cfg.theta_grid, cfg.mu_grid))
    scored = (cfg.frames_dir, size, cfg.tau_grid, named)
    s_val, x_val = part_scores(*scored, val, "validation", cfg.feature_quantile)
    chosen = thresholds(grid, s_val, [f["label"] for f in val], names)
    s_test, x_test = part_scores(*scored, test, "test", cfg.feature_quantile)
    preds = {
        v: {c: predictions(s_test[v], point, names) for c, point in chosen[v].items()}
        for v in VARIANTS
    }
    return chosen, preds, x_val, x_test


def run(cfg: GuiMasksConfig) -> dict:
    """Notation, (1)-(14): templates and features.npz written to out_dir; returns the
    report."""
    classes = load_screens(cfg.screens)
    names, size = list(classes), frame_size(cfg.frames_dir)
    check_patches(cfg, size)
    parts, (train, val, test) = split_frames(cfg)
    q = qualified(train, names, cfg.min_episodes)
    fold_of = folds([e for e, p in parts.items() if p == 0], cfg.folds, cfg.fold_seed)
    textures = load_textures_config(cfg.screens).out_dir
    named, fold_pairs = all_pairs(cfg, classes, textures, train, q, fold_of, size)
    write_templates(cfg.out_dir, named)
    x_train = train_features(
        cfg.frames_dir, size, named, fold_pairs, fold_of, train, cfg.feature_quantile
    )
    chosen, preds, x_val, x_test = recognised(cfg, size, named, val, test, names)
    x = {"train": (train, x_train), "validation": (val, x_val), "test": (test, x_test)}
    labels = names + [OTHER]
    write_features(cfg.out_dir, named, labels, x, preds)
    split = split_report(parts, [train, val, test], labels, fold_of, cfg.folds)
    report = {"split": split, "qualified": q}
    report |= {
        v: variant_report(named[v], chosen[v], preds[v], test, names) for v in VARIANTS
    }
    y_test = [f["label"] for f in test]
    return report | {"agreement_matrices": matrices(y_test, preds, names)}


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.recognition.evaluate <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        write_report(cfg.out_dir, run(cfg), TITLE)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
