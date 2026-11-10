"""EnvState and raw reset/step conversion of specs/env.md (3)-(5)."""

import dataclasses

import numpy as np
import pytest

from mcpref.runtime.env.state import EnvState
from runtime.env.fake_simulator import FRAME, raw


def test_public_state_record(world):
    state = world.reset()
    assert [f.name for f in dataclasses.fields(EnvState)] == [
        "frame",
        "line",
        "gui",
        "cls",
        "hotbar",
        "mask",
    ]
    assert isinstance(state, EnvState)
    assert state.frame.shape == (360, 640, 3) and state.frame.dtype == np.uint8
    assert state.mask.shape == (395,) and state.mask.dtype == np.bool_
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.hotbar = 2


def test_raw_reset_conversion_inventory_and_slot_order(world):
    frame = FRAME.copy()
    frame[0, 0] = [1, 2, 3]
    inventory = {
        4: {"type": "minecraft:dirt", "quantity": 64},
        0: {"type": "minecraft:stone", "quantity": 4},
        1: {"type": "minecraft:air", "quantity": 0},
        2: {"type": "mod:item", "quantity": 2},
        3: {"type": "air", "quantity": 1},
    }
    pos = {"x": 1.25, "y": -2, "z": 3.5, "yaw": 4, "pitch": -5.25}
    world.fake.reset_raw = raw(frame=frame, pos=pos, inventory=inventory)
    state = world.reset()
    assert state.frame is frame
    assert state.line == {
        "inventory": [
            {"type": "stone", "quantity": 4},
            {"type": "mod:item", "quantity": 2},
            {"type": "dirt", "quantity": 64},
        ],
        "hotbar": 0,
        "yaw": 4.0,
        "pitch": -5.25,
        "xpos": 1.25,
        "ypos": -2.0,
        "zpos": 3.5,
    }
    assert all(
        type(state.line[k]) is float for k in ("yaw", "pitch", "xpos", "ypos", "zpos")
    )
    assert (state.gui, state.cls, state.hotbar) == (False, None, 0)
    assert world.classifier.calls == []


def test_raw_step_returns_last_tick_and_ignores_reward(ready):
    final = raw(
        pos={"x": 9, "y": 8, "z": 7, "yaw": 6, "pitch": 5},
        inventory={0: {"type": "minecraft:oak_log", "quantity": 1}},
    )
    ready.fake.results.extend(raw() for _ in range(5))
    ready.fake.results.append(final)
    state = ready.step("turn_left")
    assert state.frame is final[0]["image"]
    assert state.line["inventory"] == [{"type": "oak_log", "quantity": 1}]
    assert (state.line["xpos"], state.line["ypos"], state.line["zpos"]) == (
        9.0,
        8.0,
        7.0,
    )
    assert len(ready.fake.actions) == 6


def test_raw_gui_reset_classifies_one_frame_before_mask(world):
    world.fake.reset_raw = raw(gui=True)
    state = world.reset()
    assert (state.gui, state.cls) == (True, "inventory")
    assert len(world.classifier.calls) == 1
    assert len(world.classifier.calls[0]) == 1
    assert world.classifier.calls[0][0] is world.fake.reset_raw[0]["image"]
    assert state.mask[world.index("close")]
    assert state.mask[world.index("left_click_1")]
    assert not state.mask[world.index("attack")]
