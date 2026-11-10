"""Interface config of specs/env.md: configs/runtime/env.yaml and the Delta check."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, positive, read_yaml, relative, repo_path
from mcpref.datasets.clips.config import load_config as load_clips_config

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
    raw = config_mapping(read_yaml(path), path, EnvConfig)
    for name in PATHS:
        relative(name, raw[name])
    cfg = EnvConfig(
        **raw | {"cursor_scale": positive("cursor_scale", raw["cursor_scale"])}
    )
    if load_clips_config(repo_path("clips", cfg.clips)).delta < 5:
        raise ValueError("delta must be >= 5")
    return cfg
