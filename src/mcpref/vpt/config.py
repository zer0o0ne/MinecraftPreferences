"""Notation of specs/vpt-gui-frames.md: configs/vpt-gui-frames.yaml; the access fields
shared with specs/perception-data.md; run.json and the rerun check of both."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml


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


def check_rerun(path: Path, header: dict) -> None:
    """(d) of specs/vpt-gui-frames.md, (7) of specs/perception-data.md: an existing
    run.json must hold the same value of every header key."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    changed = [
        k for k, v in header.items() if json.dumps(previous.get(k)) != json.dumps(v)
    ]
    if changed:
        raise RuntimeError(f"{path} holds other values of {changed}")


def write_run(out: Path, run: dict) -> None:
    """Behaviour: run.json, the header before planning, completed with the counts."""
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
