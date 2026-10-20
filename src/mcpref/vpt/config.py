"""Notation of specs/vpt-gui-frames.md: configs/vpt-gui-frames.yaml; the access fields
shared with specs/perception-data.md; run.json and its rerun check (d)."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from mcpref.shards.store import write_file

OUTPUT_FIELDS = ("index_urls", "min_version", "required_options", "width", "height")
OUTPUT_FIELDS += ("fps", "n_frames", "per_episode", "seed")


class Access(Protocol):
    """Interface of specs/perception-data.md: the access fields of GuiFramesConfig and
    PerceptionDataConfig, same meaning."""

    index_urls: dict[str, str]
    x_ms_version: str
    min_version: str
    required_options: dict[str, object]
    width: int
    height: int
    fetch_streams: int
    http_timeout: float
    http_retries: int
    retry_delay: float
    ffmpeg: str


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
    plan_shard_draws: int
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


def check_same(path: Path, previous: Mapping, current: Mapping) -> None:
    """(d) of specs/vpt-gui-frames.md: abort when a value of `current` differs from
    `previous` (read from `path`) as JSON text."""
    changed = [
        k for k, v in current.items() if json.dumps(previous.get(k)) != json.dumps(v)
    ]
    if changed:
        raise RuntimeError(f"{path} holds other values of {changed}")


def check_outputs(path: Path, config: dict) -> None:
    """(d) of specs/vpt-gui-frames.md: an existing run.json must hold the same output
    fields in its config, each compared as JSON text."""
    if path.exists():
        outputs = {k: config[k] for k in OUTPUT_FIELDS}
        check_same(path, json.loads(path.read_text())["config"], outputs)


def store_run(path: Path, run: dict) -> None:
    """run.json of specs/vpt-gui-frames.md, written atomically (specs/shards.md (4))."""
    write_file(path, json.dumps(run, indent=2).encode())
