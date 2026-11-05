"""Notation of specs/ppo.md: configs/ppo.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, number, positive_int


@dataclass(frozen=True)
class PpoConfig:
    """Notation: config fields."""

    horizon: int
    clip: float


def load_config(path: Path) -> PpoConfig:
    """Interface: the YAML file, validated, into PpoConfig."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PpoConfig)
    horizon = positive_int("horizon", raw["horizon"])
    clip = number("clip", raw["clip"])
    if not 0 < clip < 1:
        raise ValueError(f"clip {raw['clip']!r} is not in (0, 1)")
    return PpoConfig(horizon=horizon, clip=clip)
