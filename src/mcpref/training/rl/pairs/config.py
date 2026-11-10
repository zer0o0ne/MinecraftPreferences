"""The validated configs/training/rl/preference-pairs.yaml mapping."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import (
    config_mapping,
    integer,
    number,
    positive_int,
    read_yaml,
    relative,
)

NON_NEGATIVE = (
    "sample_states",
    "temporal_per_state",
    "nearest_per_state",
    "branch",
    "beam_width",
    "beam_depth",
)


@dataclass(frozen=True)
class RLPreferencePairsConfig:
    """Notation: all exact numbers and output paths of the block."""

    history_dir: str
    out_dir: str
    sample_states: int
    temporal_per_state: int
    nearest_per_state: int
    branch: int
    beam_width: int
    beam_depth: int
    capacity: int
    items_per_shard: int
    history_per_shard: int
    base_threshold: int | float
    temporal_penalty: int | float
    threshold_scale: int | float


def load_config(path: Path) -> RLPreferencePairsConfig:
    """Load exactly the dataclass fields and reject violated constraints."""
    raw = config_mapping(read_yaml(path), path, RLPreferencePairsConfig)
    for name in ("history_dir", "out_dir"):
        relative(name, raw[name])
    for name in NON_NEGATIVE:
        integer(name, raw[name], 0, float("inf"))
    for name in ("capacity", "items_per_shard", "history_per_shard"):
        positive_int(name, raw[name])
    for name in ("base_threshold", "threshold_scale"):
        if not 0 < number(name, raw[name]) < 1:
            raise ValueError(f"{name} {raw[name]!r} is not in (0, 1)")
    if not 0 <= number("temporal_penalty", raw["temporal_penalty"]) < float("inf"):
        raise ValueError(
            f"temporal_penalty {raw['temporal_penalty']!r} is not finite and >= 0"
        )
    return RLPreferencePairsConfig(**raw)
