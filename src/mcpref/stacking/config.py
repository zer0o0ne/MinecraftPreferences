"""Notation of specs/gui-stack.md: configs/gui-stack.yaml."""

from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml

from mcpref import REPO


@dataclass(frozen=True)
class GuiStackConfig:
    """Notation: config fields; paths resolved against the repository root."""

    masks_dir: Path
    hidden: int
    l2: float
    lr: float
    betas: tuple[float, float]
    adam_eps: float
    steps: int
    seed: int
    device: torch.device
    out_dir: Path


def mapping(raw: object, path: Path) -> dict:
    """Notation: a YAML mapping with exactly the fields of GuiStackConfig."""
    names = [f.name for f in fields(GuiStackConfig)]
    keys = raw if isinstance(raw, dict) else {}
    missing = [n for n in names if n not in keys]
    unknown = [k for k in keys if k not in names]
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")
    return keys


def repo_path(name: str, value: object) -> Path:
    """Notation: a path string, resolved against the repository root."""
    if isinstance(value, str):
        return REPO / value
    raise ValueError(f"{name} {value!r} is not a path string")


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


def device(value: object) -> torch.device:
    """Notation: a string accepted by torch.device."""
    if isinstance(value, str):
        try:
            return torch.device(value)
        except RuntimeError as error:
            raise ValueError(f"device {value!r}: {error}") from error
    raise ValueError(f"device {value!r} is not a string")


def check(cfg: GuiStackConfig) -> None:
    """Notation: the constraints on config fields."""
    for name in ("hidden", "steps", "seed"):
        if type(getattr(cfg, name)) is not int:
            raise ValueError(f"{name} {getattr(cfg, name)!r} is not an integer")
    if cfg.hidden < 1:
        raise ValueError(f"hidden {cfg.hidden} is not >= 1")
    if cfg.steps < 1:
        raise ValueError(f"steps {cfg.steps} is not >= 1")
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
    """Interface: the YAML file loaded into GuiStackConfig; violated constraints raise."""
    raw = mapping(yaml.safe_load(path.read_text()), path)
    paths = {k: repo_path(k, raw[k]) for k in ("masks_dir", "out_dir")}
    reals = {k: real(k, raw[k]) for k in ("l2", "lr", "adam_eps")}
    read = {"betas": betas(raw["betas"]), "device": device(raw["device"])}
    cfg = GuiStackConfig(**raw | paths | reals | read)
    check(cfg)
    return cfg
