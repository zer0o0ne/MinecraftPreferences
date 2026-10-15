"""`load_config` of src/mcpref/recognition/debug_overlay.py against specs/f3-overlay.md
(Notation: the config; Interface: starting values)."""

import dataclasses
from pathlib import Path

import pytest

from mcpref.recognition.debug_overlay import OverlayConfig, load_config

TEMPLATES = {"normal", "dimmed"}
VALID = {  # YAML lines of a valid config
    "screens": '"configs/gui-screens.yaml"', "prefix": '"Minecraft 1.16.5 ("',
    "text_luma": "{normal: 208.38, dimmed: 73.7}", "tolerance": "{normal: 24, dimmed: 8}",
    "box_max": "{normal: 160.1, dimmed: 63.4}", "threshold": "0.75",
}  # fmt: skip


def load(tmp_path: Path, **changes: str) -> OverlayConfig:
    path = tmp_path / "f3.yaml"
    path.write_text("".join(f"{k}: {v}\n" for k, v in (VALID | changes).items()))
    return load_config(path)


def numbers(cfg: OverlayConfig, name: str) -> dict:
    """A template mapping as a dict; its keys exactly the two templates, values floats."""
    m = getattr(cfg, name)
    assert set(m.keys()) == TEMPLATES
    assert all(type(m[k]) is float for k in TEMPLATES)
    return {k: m[k] for k in TEMPLATES}


def test_starting_values(repo, tmp_path, monkeypatch):
    """Interface: starting values; `screens` resolved against the repository root whatever
    the working directory; integers of `tolerance` read with float()."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(repo / "configs/f3-overlay.yaml")
    assert {f.name for f in dataclasses.fields(OverlayConfig)} == set(VALID)
    assert Path(repo, cfg.screens).resolve() == (repo / "configs/gui-screens.yaml")
    assert cfg.prefix == "Minecraft 1.16.5 ("
    assert numbers(cfg, "text_luma") == {"normal": 208.38, "dimmed": 73.7}
    assert numbers(cfg, "tolerance") == {"normal": 24.0, "dimmed": 8.0}
    assert numbers(cfg, "box_max") == {"normal": 160.1, "dimmed": 63.4}
    assert cfg.threshold == 0.75
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.threshold = 0.5


def test_screens_relative_to_the_repository_root(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load(tmp_path, screens='"configs/other.yaml"')
    assert Path(repo, cfg.screens).resolve() == (repo / "configs/other.yaml")


BAD = {
    "prefix empty": {"prefix": '""'},
    "prefix not a string": {"prefix": "123"},
    "prefix null": {"prefix": "null"},
    "prefix with 0x7F": {"prefix": '"Minecraft\\x7f"'},
    "prefix with a tab": {"prefix": '"Minecraft\\t1"'},
    "prefix with a newline": {"prefix": '"Minecraft\\n"'},
    "prefix beyond ASCII": {"prefix": '"Minecraft \\xe9"'},
    "text_luma a key missing": {"text_luma": "{normal: 208.38}"},
    "text_luma an extra key": {"text_luma": "{normal: 1, dimmed: 2, bright: 3}"},
    "text_luma a list": {"text_luma": "[208.38, 73.7]"},
    "text_luma a number": {"text_luma": "208.38"},
    "text_luma a boolean": {"text_luma": "{normal: true, dimmed: 73.7}"},
    "text_luma a string": {"text_luma": '{normal: "208.38", dimmed: 73.7}'},
    "text_luma null": {"text_luma": "{normal: 208.38, dimmed: null}"},
    "tolerance negative": {"tolerance": "{normal: -1, dimmed: 8}"},
    "tolerance slightly negative": {"tolerance": "{normal: 24, dimmed: -0.5}"},
    "tolerance a boolean": {"tolerance": "{normal: 24, dimmed: false}"},
    "tolerance a key missing": {"tolerance": "{dimmed: 8}"},
    "box_max a boolean": {"box_max": "{normal: true, dimmed: 63.4}"},
    "box_max a string": {"box_max": '{normal: 160.1, dimmed: "63.4"}'},
    "box_max an extra key": {"box_max": "{normal: 1, dimmed: 2, Normal: 3}"},
    "threshold 0": {"threshold": "0"},
    "threshold 0.0": {"threshold": "0.0"},
    "threshold negative": {"threshold": "-0.5"},
    "threshold above 1": {"threshold": "1.0000001"},
    "threshold 2": {"threshold": "2"},
    "threshold a boolean": {"threshold": "true"},
    "threshold a string": {"threshold": '"0.75"'},
    "threshold null": {"threshold": "null"},
}


@pytest.mark.parametrize("changes", BAD.values(), ids=BAD.keys())
def test_rejects(tmp_path, changes):
    with pytest.raises(ValueError):
        load(tmp_path, **changes)


GOOD = {  # boundaries inside the constraints
    "prefix 0x20": {"prefix": '" "'},
    "prefix 0x7E": {"prefix": '"~"'},
    "prefix 0x20-0x7E": {"prefix": '" !~Az09("'},
    "tolerance 0": {"tolerance": "{normal: 0, dimmed: 0.0}"},
    "threshold 1": {"threshold": "1"},
    "threshold 1.0": {"threshold": "1.0"},
    "threshold small": {"threshold": "0.001"},
    "text_luma unbounded": {"text_luma": "{normal: 300, dimmed: -5.5}"},
    "box_max unbounded": {"box_max": "{normal: -1, dimmed: 1000}"},
}


@pytest.mark.parametrize("changes", GOOD.values(), ids=GOOD.keys())
def test_accepts(tmp_path, changes):
    assert isinstance(load(tmp_path, **changes), OverlayConfig)


def test_numbers_read_with_float(tmp_path):
    """YAML integers of the template mappings become floats; values kept."""
    cfg = load(
        tmp_path,
        text_luma="{normal: 208, dimmed: -5.5}",
        tolerance="{normal: 0, dimmed: 8}",
        box_max="{normal: 1000, dimmed: 63}",
        threshold="1",
    )
    assert numbers(cfg, "text_luma") == {"normal": 208.0, "dimmed": -5.5}
    assert numbers(cfg, "tolerance") == {"normal": 0.0, "dimmed": 8.0}
    assert numbers(cfg, "box_max") == {"normal": 1000.0, "dimmed": 63.0}
    assert cfg.threshold == 1
