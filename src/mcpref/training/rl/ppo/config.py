"""Notation of specs/ppo.md: configs/training/rl/ppo.yaml."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, number, positive, positive_int, read_yaml


@dataclass(frozen=True)
class PpoConfig:
    """Notation: config fields."""

    horizon: int
    clip: float
    skill_eps: float
    dual_init: float
    dual_slack: float


def load_config(path: Path) -> PpoConfig:
    """Interface: the YAML file, validated, into PpoConfig."""
    raw = config_mapping(read_yaml(path), path, PpoConfig)
    horizon = positive_int("horizon", raw["horizon"])
    clip = number("clip", raw["clip"])
    if not 0 < clip < 1:
        raise ValueError(f"clip {raw['clip']!r} is not in (0, 1)")
    skill_eps = positive("skill_eps", raw["skill_eps"])
    dual_init = positive("dual_init", raw["dual_init"])
    dual_slack = positive("dual_slack", raw["dual_slack"])
    return PpoConfig(
        horizon=horizon,
        clip=clip,
        skill_eps=skill_eps,
        dual_init=dual_init,
        dual_slack=dual_slack,
    )
