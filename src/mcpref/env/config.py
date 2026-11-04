"""Interface config of specs/env.md: configs/env.yaml and the Delta check."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import config_mapping, positive, relative, repo_path
from mcpref.clips.config import load_config as load_clips_config

PATHS = ("engine", "classifier", "action_mask", "clips")


@dataclass(frozen=True)
class EnvConfig:
    """Interface: dependent config paths as written and cursor_scale = sigma."""

    engine: str
    classifier: str
    action_mask: str
    clips: str
    cursor_scale: float


def load_config(path: Path) -> EnvConfig:
    """Interface: the validated YAML mapping; clips.delta must be at least five."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, EnvConfig)
    for name in PATHS:
        relative(name, raw[name])
    cfg = EnvConfig(
        **raw | {"cursor_scale": positive("cursor_scale", raw["cursor_scale"])}
    )
    if load_clips_config(repo_path("clips", cfg.clips)).delta < 5:
        raise ValueError("delta must be >= 5")
    return cfg
