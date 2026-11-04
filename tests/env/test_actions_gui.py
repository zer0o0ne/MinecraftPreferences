"""Escape, cursor and exact click schedules of specs/env.md (4), (7), (8)."""

import numpy as np

from .fake_simulator import assert_actions, expected, raw

IDLE = expected()
CENTER = np.array([(247 - 180) / (20 / 3), (248 - 320) / (20 / 3)], dtype=np.float32)


def gui_world(world):
    world.fake.reset_raw = raw(gui=True)
    world.reset()
    return world


def test_left_click_camera_press_release_then_fully_idle(world):
    gui_world(world).step("left_click_1")
    wanted = [expected(camera=CENTER), expected(attack=1), *([IDLE] * 4)]
    assert_actions(world.fake, wanted)


def test_right_click_uses_use_only_on_press(world):
    gui_world(world).step("right_click_1")
    wanted = [expected(camera=CENTER), expected(use=1), *([IDLE] * 4)]
    assert_actions(world.fake, wanted)


def test_shift_click_holds_sneak_on_press_and_release_only(world):
    gui_world(world).step("shift_click_1")
    wanted = [
        expected(camera=CENTER),
        expected(attack=1, sneak=1),
        expected(sneak=1),
        *([IDLE] * 3),
    ]
    assert_actions(world.fake, wanted)


def test_double_click_press_release_one_three_and_two_four(world):
    gui_world(world).step("double_click_1")
    wanted = [
        expected(camera=CENTER),
        expected(attack=1),
        IDLE,
        expected(attack=1),
        IDLE,
        IDLE,
    ]
    assert_actions(world.fake, wanted)


def test_camera_only_on_first_click_tick_and_cursor_tracked(world):
    gui_world(world)
    world.step("left_click_1")
    world.step("left_click_1")
    zero = np.zeros(2, dtype=np.float32)
    wanted = [expected(camera=CENTER), expected(attack=1), *([IDLE] * 4)]
    wanted += [expected(camera=zero), expected(attack=1), *([IDLE] * 4)]
    assert_actions(world.fake, wanted)


def test_close_press_release_until_raw_false_then_idle(world):
    gui_world(world)
    world.fake.results.extend([raw(gui=True), raw()])
    state = world.step("close")
    wanted = [expected(ESC=1), expected(), *([IDLE] * 4)]
    assert_actions(world.fake, wanted)
    assert (state.gui, state.cls) == (False, None)


def test_close_repeats_pairs_beyond_delta_until_release_is_false(world):
    gui_world(world)
    world.fake.results.extend([raw(gui=True)] * 7 + [raw()])
    world.step("close")
    wanted = [expected(ESC=1 if tick % 2 == 0 else 0) for tick in range(8)]
    assert_actions(world.fake, wanted)


def test_cursor_recentred_by_raw_close_and_reopen(world):
    gui_world(world)
    world.step("left_click_1")
    world.fake.results.extend([raw(gui=True), raw()])
    world.step("close")
    world.fake.results.append(raw(gui=True))
    world.step("open_inventory")
    world.step("left_click_1")
    click = [expected(camera=CENTER), expected(attack=1), *([IDLE] * 4)]
    close = [expected(ESC=1), *([IDLE] * 5)]
    opened = [expected(inventory=1), *([IDLE] * 5)]
    assert_actions(world.fake, [*click, *close, *opened, *click])


def test_open_inventory_completes_on_raw_true_without_false_true_transition(ready):
    ready.fake.results.extend([raw()] * 5 + [raw(gui=True)])
    first = ready.step("use")
    assert (first.gui, first.cls) == (False, None)
    second = ready.step("open_inventory")
    wanted = [expected(use=1)] * 6 + [expected(inventory=1), *([IDLE] * 5)]
    assert_actions(ready.fake, wanted)
    assert (second.gui, second.cls) == (True, "inventory")
