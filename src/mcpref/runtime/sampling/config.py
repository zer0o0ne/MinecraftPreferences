"""Notation of specs/env-sampling.md: configs/runtime/sampling.yaml."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import (
    REPO,
    config_mapping,
    integer,
    number,
    positive_int,
    read_yaml,
    relative,
)
from mcpref.datasets.clips.config import load_config as load_clips_config
from mcpref.runtime.env.config import load_config as load_env_config

PATHS = ("pretrain", "env", "mcts", "perception_data", "out_dir")


@dataclass(frozen=True)
class SamplingConfig:
    """Notation: dependent paths, trajectory length, decision RNG and time margin."""

    pretrain: str
    env: str
    mcts: str
    perception_data: str
    out_dir: str
    steps: int
    seed: int
    safety_limit: float


def load_config(path: Path) -> SamplingConfig:
    """Interface: the validated YAML mapping, with a positive action-time budget."""
    raw = config_mapping(read_yaml(path), path, SamplingConfig)
    for name in PATHS:
        relative(name, raw[name])
    safety = number("safety_limit", raw["safety_limit"])
    if not safety >= 0:
        raise ValueError(f"safety_limit {raw['safety_limit']!r} is not >= 0")
    env = load_env_config(REPO / raw["env"])
    delta = load_clips_config(REPO / env.clips).delta
    if not 50 * delta - safety > 0:
        raise ValueError("50 * clips.delta - safety_limit must be > 0")
    cfg = SamplingConfig(
        **raw
        | {
            "steps": positive_int("steps", raw["steps"]),
            "seed": integer("seed", raw["seed"], 0, 2**32 - 1),
            "safety_limit": safety,
        }
    )
    object.__setattr__(cfg, "_source_path", path)
    return cfg
