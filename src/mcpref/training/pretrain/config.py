"""Notation of specs/pretrain.md: configs/training/pretrain/config.yaml."""

import math
from dataclasses import dataclass
from pathlib import Path

from mcpref.config import (
    betas,
    boolean,
    config_mapping,
    groups,
    integer,
    number,
    positive,
    positive_int,
    quantiles,
    read_yaml,
    relative,
)
from mcpref.training.devices import device

POSITIVE_INTS = ("epochs", "batch", "micro_batch", "checkpoint_steps")
POSITIVE_INTS += ("keep_checkpoints", "diagnostic_window", "reference_steps")
POSITIVE_INTS += ("reference_pairs", "calibration_bins")
POSITIVE_REALS = ("w_pi", "w_P", "grad_clip", "adam_eps", "latent_init_scale")
CONFIGS = ("perception", "action_mask", "policy", "preference", "diagnostics")
PATHS = ("index_dir", "out_dir", *CONFIGS)


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
    retrain: bool
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
    raw = config_mapping(read_yaml(path), path, PretrainConfig)
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
    read |= {"retrain": boolean("retrain", raw["retrain"])}
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
