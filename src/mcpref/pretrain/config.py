"""Notation of specs/pretrain.md: configs/pretrain.yaml."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, integer, number, positive_int, relative
from mcpref.training.devices import device

POSITIVE_INTS = ("epochs", "batch", "micro_batch", "checkpoint_steps")
POSITIVE_INTS += ("keep_checkpoints",)
POSITIVE_REALS = ("w_pi", "w_P", "grad_clip", "adam_eps")
GROUP_KEYS = {"lr", "weight_decay"}


@dataclass(frozen=True)
class PretrainConfig:
    """Notation: config fields as written; paths relative to the repository root."""

    index_dir: str
    out_dir: str
    seed: int
    epochs: int
    batch: int
    micro_batch: int
    w_pi: float
    w_P: float
    groups: dict[str, dict[str, float]]
    betas: tuple[float, float]
    adam_eps: float
    warmup_steps: int
    lr_floor: float
    grad_clip: float
    checkpoint_steps: int
    keep_checkpoints: int
    device: str


def beta(value: object) -> float:
    """Notation: one of `betas`, a `number` in [0, 1)."""
    x = number("betas", value)
    if not 0 <= x < 1:
        raise ValueError(f"betas {value!r} is not in [0, 1)")
    return x


def positive(name: str, value: object) -> float:
    """Notation: a `number` > 0."""
    x = number(name, value)
    if not x > 0:
        raise ValueError(f"{name} {value!r} is not > 0")
    return x


def betas(value: object) -> tuple[float, float]:
    """Notation: a list of exactly two numbers, each in [0, 1)."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"betas {value!r} are not a list of two")
    return beta(value[0]), beta(value[1])


def group(name: object, value: object) -> dict[str, float]:
    """Notation: a group: exactly `lr` (> 0) and `weight_decay` (>= 0)."""
    if type(name) is not str or not isinstance(value, dict) or set(value) != GROUP_KEYS:
        raise ValueError(f"group {name!r}: {value!r} is not lr and weight_decay")
    decay = number("weight_decay", value["weight_decay"])
    if not decay >= 0:
        raise ValueError(f"group {name}: weight_decay {decay} is not >= 0")
    return {"lr": positive("lr", value["lr"]), "weight_decay": decay}


def groups(value: object) -> dict[str, dict[str, float]]:
    """Notation: a mapping from string names to groups, in config order."""
    if type(value) is not dict:
        raise ValueError(f"groups {value!r} is not a mapping")
    return {name: group(name, g) for name, g in value.items()}


def load_config(path: Path) -> PretrainConfig:
    """Interface: the YAML file, validated, into PretrainConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PretrainConfig)
    for k in ("index_dir", "out_dir"):
        relative(k, raw[k])
    device(raw["device"])
    read = {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read |= {"seed": integer("seed", raw["seed"], 0, 2**32 - 1)}
    read |= {"warmup_steps": integer("warmup_steps", raw["warmup_steps"], 0, math.inf)}
    lr_floor = number("lr_floor", raw["lr_floor"])
    if not 0 <= lr_floor <= 1:
        raise ValueError(f"lr_floor {lr_floor} is not in [0, 1]")
    read |= {"lr_floor": lr_floor, "betas": betas(raw["betas"])}
    return PretrainConfig(**raw | read | {"groups": groups(raw["groups"])})
