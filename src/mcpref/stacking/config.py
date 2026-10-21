"""Notation of specs/gui-stack.md: configs/gui-stack.yaml."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from mcpref import config_mapping, positive_int, repo_path
from mcpref.training.devices import device

POSITIVE = ("hidden", "epochs", "batch", "checkpoint_steps")


@dataclass(frozen=True)
class GuiStackConfig:
    """Notation: config fields; paths resolved against the repository root."""

    masks_dir: Path
    hidden: int
    l2: float
    lr: float
    betas: tuple[float, float]
    adam_eps: float
    epochs: int
    batch: int
    checkpoint_steps: int
    seed: int
    device: torch.device
    out_dir: Path


def real(name: str, value: object) -> float:
    """Notation: a number read with float()."""
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} {value!r} is not read by float()") from error


def betas(value: object) -> tuple[float, float]:
    """Notation: exactly two betas, read with float()."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"betas {value!r} are not a list of two")
    return real("betas", value[0]), real("betas", value[1])


def check(cfg: GuiStackConfig) -> None:
    """Notation: the constraints on config fields."""
    if type(cfg.seed) is not int:
        raise ValueError(f"seed {cfg.seed!r} is not an integer")
    if not 0 <= cfg.seed < 2**64:
        raise ValueError(f"seed {cfg.seed} is not in [0, 2^64)")
    if not cfg.l2 >= 0:
        raise ValueError(f"l2 {cfg.l2} is not >= 0")
    if not cfg.lr > 0:
        raise ValueError(f"lr {cfg.lr} is not > 0")
    if not all(0 <= b < 1 for b in cfg.betas):
        raise ValueError(f"betas {cfg.betas} are not in [0, 1)")
    if not cfg.adam_eps > 0:
        raise ValueError(f"adam_eps {cfg.adam_eps} is not > 0")


def load_config(path: Path) -> GuiStackConfig:
    """Interface: the YAML file as a GuiStackConfig; violated constraints raise."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, GuiStackConfig)
    paths = {k: repo_path(k, raw[k]) for k in ("masks_dir", "out_dir")}
    reals = {k: real(k, raw[k]) for k in ("l2", "lr", "adam_eps")}
    ints = {k: positive_int(k, raw[k]) for k in POSITIVE}
    read = {"betas": betas(raw["betas"]), "device": device(raw["device"])}
    cfg = GuiStackConfig(**raw | paths | reals | ints | read)
    check(cfg)
    return cfg
