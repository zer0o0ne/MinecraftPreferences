"""Notation: the validated 22-field configs/rl-cycle.yaml mapping."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, integer, number, positive, positive_int, relative
from mcpref.pretrain.config import betas, groups
from mcpref.training.devices import device

PATHS = ("sampling", "ppo", "rl_pairs", "out_dir")
POSITIVE_INTS = (
    "epochs",
    "batch",
    "micro_batch",
    "items_per_shard",
    "checkpoint_steps",
    "keep_checkpoints",
)
POSITIVE_REALS = ("w_ppo", "w_delta", "w_metra", "w_P", "adam_eps", "grad_clip")


@dataclass(frozen=True)
class RLCycleConfig:
    """All paths, exact training numbers and declared optimizer groups of one cycle."""

    sampling: str
    ppo: str
    rl_pairs: str
    out_dir: str
    seed: int
    epochs: int
    batch: int
    micro_batch: int
    items_per_shard: int
    checkpoint_steps: int
    keep_checkpoints: int
    w_ppo: float
    w_delta: float
    w_metra: float
    w_P: float
    adam_eps: float
    grad_clip: float
    groups: dict[str, dict[str, float]]
    betas: tuple[float, float]
    warmup_steps: int
    lr_floor: float
    device: str


def load_config(path: Path) -> RLCycleConfig:
    """Load exactly the dataclass fields and reject every violated constraint."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, RLCycleConfig)
    for name in PATHS:
        relative(name, raw[name])
    read = {name: positive_int(name, raw[name]) for name in POSITIVE_INTS}
    read |= {name: positive(name, raw[name]) for name in POSITIVE_REALS}
    read |= {
        "seed": integer("seed", raw["seed"], 0, 2**32 - 1),
        "warmup_steps": integer("warmup_steps", raw["warmup_steps"], 0, math.inf),
        "betas": betas(raw["betas"]),
        "groups": groups(raw["groups"]),
    }
    floor = number("lr_floor", raw["lr_floor"])
    if not 0 <= floor <= 1:
        raise ValueError(f"lr_floor {raw['lr_floor']!r} is not in [0, 1]")
    device(raw["device"])
    return RLCycleConfig(**raw | read | {"lr_floor": floor})
