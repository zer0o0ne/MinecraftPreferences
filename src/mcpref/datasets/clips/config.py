"""Notation of specs/clips.md: configs/datasets/clips.yaml."""

from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, positive_int, read_yaml, repo_path


@dataclass(frozen=True)
class ClipsConfig:
    """Notation: delta = Delta, kept lines per interval; actions = the path of
    configs/datasets/vpt/actions.yaml, resolved against the repository root."""

    delta: int
    actions: Path


def load_config(path: Path) -> ClipsConfig:
    """Interface: the YAML file loaded into ClipsConfig; violated constraints raise."""
    raw = config_mapping(read_yaml(path), path, ClipsConfig)
    return ClipsConfig(
        positive_int("delta", raw["delta"]), repo_path("actions", raw["actions"])
    )
