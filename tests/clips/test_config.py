"""`load_config` and the starting values of specs/clips.md (Notation, Interface)."""

import dataclasses
from pathlib import Path

import pytest

from mcpref.clips.config import ClipsConfig, load_config
from mcpref.vpt.actions import ActionsConfig
from mcpref.vpt.actions import load_config as load_actions

ACTIONS = "configs/vpt-actions.yaml"


def write(directory: Path, text: str) -> Path:
    path = directory / "clips.yaml"
    path.write_text(text)
    return path


def test_load_config_reads_every_field(tmp_path):
    cfg = load_config(write(tmp_path, f"delta: 7\nactions: {ACTIONS}\n"))
    assert isinstance(cfg, ClipsConfig)
    assert [f.name for f in dataclasses.fields(cfg)] == ["delta", "actions"]
    assert cfg.delta == 7 and type(cfg.delta) is int
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.delta = 1


def test_actions_is_resolved_against_the_repository_root(tmp_path, repo, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(write(tmp_path, "delta: 1\nactions: configs/other.yaml\n"))
    assert Path(cfg.actions).is_absolute()
    assert Path(cfg.actions).resolve() == (repo / "configs/other.yaml").resolve()


def test_actions_config_loads_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(write(tmp_path, f"delta: 20\nactions: {ACTIONS}\n"))
    assert load_actions(Path(cfg.actions)) == ActionsConfig(turn_deg=10.0, r_min=3)


@pytest.mark.parametrize(
    "value", ["0", "-1", "true", "false", "20.0", "1.5", "'20'", "null"]
)
def test_delta_must_be_an_integer_of_at_least_1(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, f"delta: {value}\nactions: {ACTIONS}\n"))


def test_delta_of_1_is_accepted(tmp_path):
    assert load_config(write(tmp_path, f"delta: 1\nactions: {ACTIONS}\n")).delta == 1


def test_approved_config_has_starting_values(repo):
    cfg = load_config(repo / "configs/clips.yaml")
    assert cfg.delta == 20
    assert Path(cfg.actions).resolve() == (repo / ACTIONS).resolve()
