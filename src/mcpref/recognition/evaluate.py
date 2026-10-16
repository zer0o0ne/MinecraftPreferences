"""Config and `main` of specs/gui-masks.md: the run (R), (I), (T), (S), (E) in order."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.label_files import write_json
from mcpref.gui.screens import load_screens
from mcpref.gui.textures import load_textures_config
from mcpref.recognition.frame_set import checks, read_state, write_index_shards
from mcpref.recognition.pairs import b_shards, c_shards, catalogue_shards
from mcpref.recognition.recognise import score
from mcpref.recognition.report import report, write_report
from mcpref.recognition.stored import load_model

PROG = "mcpref.recognition.evaluate"
TITLE = "GUI masks report"


@dataclass(frozen=True)
class GuiMasksConfig:
    """Notation: configs/gui-masks.yaml; paths relative to the repository root, kept as
    written in the config (messages of (R)) and resolved against the root where used."""

    frames_dir: str
    labels_dir: str
    screens: str
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
    out_dir: str


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
    if type(cfg.chunk_rows) is not int or cfg.chunk_rows < 2 or cfg.chunk_rows % 2:
        raise ValueError(f"chunk_rows {cfg.chunk_rows} is not an even integer >= 2")
    if type(cfg.folds) is not int or cfg.folds < 2:
        raise ValueError(f"folds {cfg.folds} is not an integer >= 2")
    if type(cfg.fold_seed) is not int:
        raise ValueError(f"fold_seed {cfg.fold_seed} is not an integer")
    if not 0 < cfg.feature_quantile <= 1:
        raise ValueError(f"feature_quantile {cfg.feature_quantile} is not in (0, 1]")


def read_mapping(path: Path) -> dict:
    """(R): the config file's mapping as written."""
    return yaml.safe_load(path.read_text())


def load_config(path: Path) -> GuiMasksConfig:
    """Interface: the YAML file loaded into GuiMasksConfig; violated constraints raise."""
    raw = read_mapping(path)
    exact = {k: decimals(raw[k]) for k in ("split", "theta_grid", "mu_grid")}
    cfg = GuiMasksConfig(**raw | exact | {"tau_grid": tuple(raw["tau_grid"])})
    check(cfg)
    return cfg


def run(cfg: GuiMasksConfig, path: Path) -> None:
    """Behaviour: (R), (I), (T), (S), (E) in order; run.json complete at the end."""
    raw, out = read_mapping(path), REPO / cfg.out_dir
    header = checks(cfg, path, raw)
    classes = load_screens(REPO / cfg.screens)
    names = list(classes)
    write_index_shards(cfg, names, header["frames"])
    state = read_state(cfg, names, header["frames"]["shards"])
    textures = load_textures_config(REPO / cfg.screens).out_dir
    catalogue_shards(out, classes, textures, state.size)
    b_shards(cfg, state)
    c_shards(cfg, state)
    model = load_model(out, classes, state.q, cfg.folds)
    chosen = score(cfg, state, model, raw)
    write_report(out, report(cfg, state, model, chosen), TITLE)
    write_json(out / "run.json", header | {"complete": True})


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.recognition.evaluate <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.config)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
