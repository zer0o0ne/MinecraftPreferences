"""`load_config` and the starting values of specs/vpt-gui-frames.md (Notation, Interface)."""

import dataclasses
from pathlib import Path

import pytest
import tomli
import yaml
from vpt_world import FIELDS

from mcpref.vpt.gui_frames import GuiFramesConfig, load_config

NAMES = ["index_urls", "x_ms_version", "min_version", "required_options", "width", "height",
         "fps", "n_frames", "per_episode", "seed", "workers", "http_timeout", "http_retries",
         "retry_delay", "ffmpeg", "ffmpeg_rw_timeout", "ffmpeg_timeout", "verify_relpath",
         "verify_frames", "out_dir"]  # fmt: skip
BLOB = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/"
STARTING = {
    "index_urls": {"6xx": BLOB + "all_6xx_Jun_29.json", "7xx": BLOB + "all_7xx_Apr_6.json",
                   "8xx": BLOB + "all_8xx_Jun_29.json", "9xx": BLOB + "all_9xx_Jun_29.json",
                   "10xx": BLOB + "all_10xx_Jun_29.json"},
    "x_ms_version": "2019-12-12", "min_version": "6.12",
    "required_options": {"guiScale": 2, "fullscreen": False, "framebufferWidth": 1280,
                         "framebufferHeight": 720, "lang": "en_us", "forceUnicodeFont": False,
                         "resourcePacks": "[]"},
    "width": 640, "height": 360, "fps": 20, "n_frames": 5000, "per_episode": 5, "seed": 0,
    "workers": 8, "http_timeout": 60, "http_retries": 3, "retry_delay": 10,
    "ffmpeg": "/usr/bin/ffmpeg", "ffmpeg_rw_timeout": 60, "ffmpeg_timeout": 300,
    "verify_relpath": "data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4",
    "verify_frames": [0, 127, 128, 129, 1271, 5999, 6000],
}  # fmt: skip


def test_load_config_reads_every_field(tmp_path):
    fields = FIELDS | {"index_urls": {"7xx": "http://h/i7", "6xx": "http://h/i6"},
                       "ffmpeg": "/x/ffmpeg", "out_dir": "data/elsewhere"}  # fmt: skip
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(fields, sort_keys=False))
    cfg = load_config(path)
    assert isinstance(cfg, GuiFramesConfig)
    assert [f.name for f in dataclasses.fields(cfg)] == NAMES
    assert {k: getattr(cfg, k) for k in NAMES if k != "out_dir"} == {
        k: v for k, v in fields.items() if k != "out_dir"
    }
    assert list(cfg.index_urls) == ["7xx", "6xx"]  # Notation: subsets in config order
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1


def test_out_dir_is_relative_to_repository_root(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(FIELDS | {"index_urls": {}, "ffmpeg": "f", "out_dir": "a/b"})
    )
    assert repo / load_config(path).out_dir == repo / "a/b"


def test_approved_config_has_starting_values(repo):
    cfg = load_config(repo / "configs/vpt-gui-frames.yaml")
    assert {k: getattr(cfg, k) for k in STARTING} == STARTING
    assert list(cfg.index_urls) == ["6xx", "7xx", "8xx", "9xx", "10xx"]
    assert Path(repo, cfg.out_dir) == repo / "data/gui_frames"


def test_pyproject_pins_httpx(repo):
    project = tomli.loads((repo / "pyproject.toml").read_text())["project"]
    assert "httpx==0.28.1" in project["dependencies"]
