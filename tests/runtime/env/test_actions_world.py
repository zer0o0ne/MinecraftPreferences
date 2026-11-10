"""Exact low-level dictionaries of specs/env.md (5)-(7), (9): movement, turns, hotbar, inventory."""

import numpy as np
import pytest

from runtime.env.fake_simulator import assert_actions, expected, raw

IDLE = expected()
MOVES = {
    "forward": {"forward": 1},
    "back": {"back": 1},
    "left": {"left": 1},
    "right": {"right": 1},
    "forward_left": {"forward": 1, "left": 1},
    "forward_right": {"forward": 1, "right": 1},
    "back_left": {"back": 1, "left": 1},
    "back_right": {"back": 1, "right": 1},
}


def idle(n):
    return [IDLE for _ in range(n)]


def camera(pitch, yaw):
    return np.array([pitch, yaw], dtype=np.float32)


@pytest.mark.parametrize("name", list(MOVES))
def test_movement_held_for_all_six_ticks(ready, name):
    ready.step(name)
    assert_actions(ready.fake, [expected(**MOVES[name])] * 6)


@pytest.mark.parametrize("name", list(MOVES))
def test_jump_movement_has_jump_and_movement_keys(ready, name):
    ready.step("jump_" + name)
    action = expected(**MOVES[name], jump=1)
    assert_actions(ready.fake, [action] * 6)


def test_jump_and_sneak_dictionaries(ready):
    ready.step("jump")
    assert_actions(ready.fake, [expected(jump=1)] * 6)


def test_sneak_dictionary(ready):
    ready.step("sneak")
    assert_actions(ready.fake, [expected(sneak=1)] * 6)


@pytest.mark.parametrize(
    "name, delta",
    [("turn_left", (0, -10)), ("turn_right", (0, 10)),
     ("turn_up", (-10, 0)), ("turn_down", (10, 0))],
)  # fmt: skip
def test_turn_only_on_first_tick(ready, name, delta):
    ready.step(name)
    assert_actions(ready.fake, [expected(camera=camera(*delta)), *idle(5)])


@pytest.mark.parametrize("slot", range(2, 10))
def test_hotbar_presses_one_tick_and_updates_wrapper_state(ready, slot):
    state = ready.step(f"hotbar_{slot}")
    assert_actions(ready.fake, [expected(**{f"hotbar.{slot}": 1}), *idle(5)])
    assert state.hotbar == slot - 1 and state.line["hotbar"] == slot - 1
    assert not state.mask[ready.index(f"hotbar_{slot}")]


def test_current_hotbar_is_masked_and_idles(ready):
    state = ready.step("hotbar_1")
    assert_actions(ready.fake, idle(6))
    assert state.hotbar == 0


def test_masked_gui_action_idles_exactly_delta(ready):
    ready.step("close")
    assert_actions(ready.fake, idle(6))


def test_open_inventory_without_signal_holds_at_most_delta(ready):
    state = ready.step("open_inventory")
    assert_actions(ready.fake, [expected(inventory=1)] * 6)
    assert (state.gui, state.cls) == (False, None)


def test_open_inventory_completes_on_raw_true_then_idles(ready):
    ready.fake.results.extend([raw(), raw(gui=True)])
    state = ready.step("open_inventory")
    assert_actions(ready.fake, [expected(inventory=1), expected(inventory=1), *idle(4)])
    assert (state.gui, state.cls) == (True, "inventory")
