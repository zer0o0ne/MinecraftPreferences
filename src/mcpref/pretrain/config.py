"""Notation of specs/pretrain.md: configs/pretrain.yaml."""

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from mcpref import (
    config_mapping,
    fraction,
    integer,
    number,
    positive,
    positive_int,
    relative,
)
from mcpref.training.devices import device

POSITIVE_INTS = ("epochs", "batch", "micro_batch", "checkpoint_steps")
POSITIVE_INTS += ("keep_checkpoints", "diagnostic_window", "reference_steps")
POSITIVE_INTS += ("reference_pairs", "calibration_bins")
POSITIVE_REALS = ("w_pi", "w_P", "grad_clip", "adam_eps", "latent_init_scale")
CONFIGS = ("perception", "action_mask", "policy", "preference", "diagnostics")
PATHS = ("index_dir", "out_dir", *CONFIGS)
GROUP_KEYS = {"lr", "weight_decay"}


@dataclass(frozen=True)
class PretrainConfig:
    """Notation: config fields as written; paths relative to the repository root."""

    index_dir: str
    out_dir: str
    perception: str
    action_mask: str
    policy: str
    preference: str
    diagnostics: str
    seed: int
    epochs: int
    batch: int
    micro_batch: int
    diagnostic_window: int
    reference_steps: int
    reference_pairs: int
    calibration_bins: int
    saturation_threshold: float
    inc_classification_threshold: float
    quantiles: tuple[float, ...]
    action_copy_eps: float
    latent_init_scale: float
    w_phi: float
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


def betas(value: object) -> tuple[float, float]:
    """Notation: a list of exactly two numbers, each in [0, 1)."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"betas {value!r} are not a list of two")
    return fraction("betas", value[0]), fraction("betas", value[1])


def quantiles(value: object) -> tuple[float, ...]:
    """(R): distinct quantiles strictly increasing in (0, 1)."""
    if not isinstance(value, list) or not value:
        raise ValueError("quantiles must be a non-empty list")
    result = tuple(number("quantiles", item) for item in value)
    if any(not 0 < q < 1 for q in result) or any(b <= a for a, b in pairwise(result)):
        raise ValueError("quantiles must be distinct and strictly increasing in (0, 1)")
    return result


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


def diagnostic_values(raw: dict) -> dict:
    """(R): validated diagnostic thresholds, quantiles and copy epsilon."""
    saturation = number("saturation_threshold", raw["saturation_threshold"])
    if not 0 < saturation < 0.5:
        raise ValueError("saturation_threshold must be in (0, 1/2)")
    inc_threshold = number(
        "inc_classification_threshold", raw["inc_classification_threshold"]
    )
    if not 0 < inc_threshold < 1:
        raise ValueError("inc_classification_threshold must be in (0, 1)")
    copy_eps = number("action_copy_eps", raw["action_copy_eps"])
    if not 0 <= copy_eps <= 1:
        raise ValueError("action_copy_eps must be in [0, 1]")
    return {
        "betas": betas(raw["betas"]),
        "quantiles": quantiles(raw["quantiles"]),
        "saturation_threshold": saturation,
        "inc_classification_threshold": inc_threshold,
        "action_copy_eps": copy_eps,
    }


def load_config(path: Path) -> PretrainConfig:
    """Interface: the YAML file, validated, into PretrainConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PretrainConfig)
    for k in PATHS:
        relative(k, raw[k])
    device(raw["device"])
    read = {k: positive_int(k, raw[k]) for k in POSITIVE_INTS}
    read |= {k: positive(k, raw[k]) for k in POSITIVE_REALS}
    read["w_phi"] = number("w_phi", raw["w_phi"])
    if not read["w_phi"] >= 0:
        raise ValueError("w_phi must be >= 0")
    if not math.isfinite(read["latent_init_scale"]):
        raise ValueError("latent_init_scale must be finite")
    read |= {"seed": integer("seed", raw["seed"], 0, 2**32 - 1)}
    read |= {"warmup_steps": integer("warmup_steps", raw["warmup_steps"], 0, math.inf)}
    lr_floor = number("lr_floor", raw["lr_floor"])
    if not 0 <= lr_floor <= 1:
        raise ValueError(f"lr_floor {lr_floor} is not in [0, 1]")
    read |= {"lr_floor": lr_floor} | diagnostic_values(raw)
    if read["checkpoint_steps"] % read["diagnostic_window"]:
        raise ValueError("checkpoint_steps % diagnostic_window must be 0")
    if read["reference_steps"] % read["checkpoint_steps"]:
        raise ValueError("reference_steps % checkpoint_steps must be 0")
    return PretrainConfig(**raw | read | {"groups": groups(raw["groups"])})
