"""Notation of specs/vpt-gui-frames.md: configs/vpt-gui-frames.yaml."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GuiFramesConfig:
    """Notation: config fields; `out_dir` relative to the repository root."""

    index_urls: dict[str, str]
    x_ms_version: str
    min_version: str
    required_options: dict[str, object]
    width: int
    height: int
    fps: int
    n_frames: int
    per_episode: int
    seed: int
    workers: int
    fetch_streams: int
    http_timeout: float
    http_retries: int
    retry_delay: float
    ffmpeg: str
    ffmpeg_rw_timeout: float
    ffmpeg_timeout: float
    verify_relpath: str
    verify_frames: list[int]
    out_dir: str


def load_config(path: Path) -> GuiFramesConfig:
    """Interface: the YAML file loaded into GuiFramesConfig."""
    return GuiFramesConfig(**yaml.safe_load(path.read_text()))
