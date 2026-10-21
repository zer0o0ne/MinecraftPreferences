"""`clip_masks` of specs/action-mask.md (4) on a hand-written clip: `info` as clip.json of
specs/perception-data.md, `lines` from its first state's line with cut lines inside."""

import numpy as np
import pytest
from mask_oracle import (
    LIMIT,
    WORLD,
    allowed,
    death_allowed,
    gui_allowed,
    names,
    world_allowed,
)

from mcpref.gui.mask import clip_masks

A = names(90)
FIRST = 200
STATES = [200, 204, 208, 215, 219, 223]  # lines 209-211 cut between states 2 and 3
HOTBAR = [3, 0, 5, 1, 7, 2]  # h of each state's line; every other line has 8


def clip() -> tuple[dict, list[dict]]:
    """States: world at -90; crafting_book; death; the first line of a container (GUI flag
    false, `isGuiOpen` true) at 90; chest3; world. Lines carry their own pitch and flag."""
    info = {
        "id": "6.13_x_200", "states": STATES, "gui": [False, True, True, False, True, False],
        "classes": [None, "crafting_book", "death", None, "chest3", None],
        "pitch": [-90.0, 12.5, 90.0, 90.0, -3.0, 45.0],
    }  # fmt: skip
    lines = [{"hotbar": 8, "pitch": 0.0, "isGuiOpen": False} for _ in range(24)]
    for line, h in zip(STATES, HOTBAR, strict=True):
        lines[line - FIRST]["hotbar"] = h
    lines[215 - FIRST]["isGuiOpen"] = True
    return info, lines


def test_rows_are_the_states_masks(masks, classes):
    info, lines = clip()
    found = clip_masks(masks, info, lines)
    assert found.dtype == np.bool_ and found.shape == (6, 395)
    expected = [
        world_allowed(-90.0, 3, LIMIT),
        gui_allowed(classes["crafting_book"]["elements"]),
        death_allowed(classes["death"]["elements"]),
        world_allowed(90.0, 1, LIMIT),
        gui_allowed(classes["chest3"]["elements"]),
        world_allowed(45.0, 2, LIMIT),
    ]
    assert [allowed(row, A) for row in found] == expected


def test_world_rows_by_hand(masks):
    # h from lines[states[s] - states[0]], theta from info["pitch"]
    found = clip_masks(masks, *clip())
    world = set(WORLD)
    assert allowed(found[0], A) == world - {"turn_up", "hotbar_4"}
    assert allowed(found[3], A) == world - {"turn_down", "hotbar_2"}
    assert allowed(found[5], A) == world - {"hotbar_3"}


def test_one_state_clip(masks):
    info = {"states": [7], "gui": [True], "classes": ["sleep"], "pitch": [0.0]}
    found = clip_masks(masks, info, [{"hotbar": 4}])
    assert found.shape == (1, 395)
    assert allowed(found[0], A) == {"close", "left_click_37", "shift_click_37"}


@pytest.mark.parametrize("cls", ["pause", "other"])
def test_state_outside_the_domain_raises(masks, cls):
    info, lines = clip()
    info["classes"][4] = cls
    with pytest.raises(ValueError) as error:
        clip_masks(masks, info, lines)
    assert str(error.value) == f"no mask for the screen {cls}"
