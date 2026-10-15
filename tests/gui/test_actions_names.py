"""`gui_action_names`, `action_names` and the config of specs/vpt-gui-actions.md
(Interface)."""

import dataclasses

import pytest
from gui_action_lines import SCREENS

from mcpref.gui.actions import (
    GuiActionsConfig,
    action_names,
    gui_action_names,
    load_config,
)
from mcpref.gui.screens import load_screens
from mcpref.vpt.actions import WORLD_ACTIONS

KINDS = ("left_click", "right_click", "shift_click", "double_click")


def names(n_max: int) -> tuple[str, ...]:
    return ("close", *(f"{k}_{i}" for k in KINDS for i in range(1, n_max + 1)))


def test_gui_action_names_of_the_tiny_catalogue():
    assert gui_action_names(SCREENS) == names(40)
    assert action_names(SCREENS) == WORLD_ACTIONS + names(40)


def test_n_max_is_the_largest_index_of_the_catalogue():
    lectern_max = {k: v for k, v in SCREENS.items() if k != "box"}
    assert gui_action_names(lectern_max) == names(38)


def test_names_of_the_approved_catalogue(repo):
    screens = load_screens(repo / "configs/gui-screens.yaml")
    found = gui_action_names(screens)
    assert found == names(90) and len(found) == 361
    assert found[:3] == ("close", "left_click_1", "left_click_2")
    assert found[-1] == "double_click_90"
    assert action_names(screens) == WORLD_ACTIONS + found
    assert len(WORLD_ACTIONS) == 34 and len(action_names(screens)) == 395


def test_approved_config(repo):
    cfg = load_config(repo / "configs/vpt-gui-actions.yaml")
    assert cfg == GuiActionsConfig(double_click_ms=250)
    assert [f.name for f in dataclasses.fields(cfg)] == ["double_click_ms"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.double_click_ms = 1


@pytest.mark.parametrize("text, value", [("1", 1), ("7", 7), ("250", 250)])
def test_config_reads_an_integer(tmp_path, text, value):
    path = tmp_path / "c.yaml"
    path.write_text(f"double_click_ms: {text}\n")
    found = load_config(path).double_click_ms
    assert found == value and type(found) is int


@pytest.mark.parametrize("text", ["true", "false", "0", "-1", "250.0", "'250'", "null"])
def test_config_rejects(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(f"double_click_ms: {text}\n")
    with pytest.raises(ValueError):
        load_config(path)
