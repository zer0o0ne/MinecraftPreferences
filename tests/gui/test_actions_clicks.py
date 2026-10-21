"""(3) clicks and (4) drags of specs/vpt-gui-actions.md; hand cases, tiny catalogue."""

import pytest
from gui_action_lines import (
    B37,
    MS,
    OUT,
    S1,
    S2,
    S10,
    SCREENS,
    episode,
    inv,
    px,
    world,
)

from mcpref.gui.actions import GuiActions, gui_actions
from mcpref.vpt.actions import Segment as S

fs = frozenset


def run(lines: list[dict], classes: dict) -> GuiActions:
    return gui_actions(lines, classes, SCREENS, MS)


def test_left_right_and_button_clicks():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[1], at=S2),
                    inv(press=[0], at=B37), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("right_click_2", 3, 3), S("left_click_37", 4, 4)]
    assert run(lines, {2: "inv", 3: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs())


@pytest.mark.parametrize(
    "key, b, at, action",
    [
        ("left.shift", 1, S2, "shift_click_2"),  # shift + right on a slot
        ("right.shift", 0, S1, "shift_click_1"),
        ("right.shift", 1, S10, "shift_click_10"),
        ("left.shift", 0, B37, "shift_click_37"),  # on a button (b37: right ignored)
        ("left.control", 0, S1, "left_click_1"),  # not a shift key
    ],
)
def test_shift_click(key, b, at, action):
    lines = episode(world(), inv(), inv(keys=[key], press=[b], at=at), world())
    assert run(lines, {2: "inv"}) == GuiActions([S(action, 2, 2)], fs(), fs())


@pytest.mark.parametrize("keys", [[], ["left.shift"]])
@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize(
    "cls, at", [("inv", OUT), ("other", S1), ("pause", S1), ("death", S1)]
)
def test_press_without_element_is_foreign(keys, b, cls, at):
    # `pause` and `death` have button 37 under S1: still no element (Notation e_t)
    lines = episode(world(), inv(), inv(), inv(keys=keys, press=[b], at=at), world())
    assert run(lines, {2: "inv", 3: cls}) == GuiActions([], fs(), fs({3}))


def test_middle_button_is_ignored():
    lines = episode(world(), inv(), inv(press=[2], at=OUT), inv(press=[2], at=S1),
                    world())  # fmt: skip
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


@pytest.mark.parametrize(
    "x, y, action",
    [
        (199, 218, None),  # frame x 99.5: left of slot 1 [100, 118)
        (200, 218, "left_click_1"),
        (235, 218, "left_click_1"),  # 117.5
        (236, 218, "left_click_2"),  # 118
        (218, 199, None),
        (218, 235, "left_click_1"),
        (218, 236, "left_click_10"),  # slot 10 at y [118, 136)
    ],
)
def test_press_position_is_halved(x, y, action):
    lines = episode(world(), inv(), inv(press=[0], at=(float(x), float(y))), world())
    clicked = GuiActions([S(action, 2, 2)], fs(), fs())
    assert run(lines, {2: "inv"}) == (
        clicked if action else GuiActions([], fs(), fs({2}))
    )


def test_drag_across_two_slots_is_foreign():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(hold=[0], at=S2),
                    inv(at=S2), world())  # fmt: skip
    expected = GuiActions([S("left_click_1", 2, 2)], fs(), fs({2, 3}))
    assert run(lines, {2: "inv"}) == expected


def test_right_drag_is_foreign():
    lines = episode(world(), inv(), inv(press=[1], at=S1), inv(hold=[1], at=S1),
                    inv(hold=[1], at=S2), inv(), world())  # fmt: skip
    expected = GuiActions([S("right_click_1", 2, 2)], fs(), fs({2, 3, 4}))
    assert run(lines, {2: "inv"}) == expected


@pytest.mark.parametrize("away", [S2, OUT])
def test_hold_leaving_and_coming_back_is_foreign(away):
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(hold=[0], at=away),
                    inv(hold=[0], at=S1), inv(at=S1), world())  # fmt: skip
    expected = GuiActions([S("left_click_1", 2, 2)], fs(), fs({2, 3, 4}))
    assert run(lines, {2: "inv"}) == expected


def test_hold_inside_one_slot():
    lines = episode(world(), inv(), inv(press=[0], at=px(100, 100)),
                    inv(hold=[0], at=px(117.5, 117.5)), inv(hold=[0], at=px(108, 101)),
                    inv(at=S2), world())  # fmt: skip
    assert run(lines, {2: "inv"}) == GuiActions([S("left_click_1", 2, 2)], fs(), fs())


@pytest.mark.parametrize("cls", ["lectern", "other", "pause", "death"])
def test_hold_uses_the_class_of_its_press(cls):
    # e_r with c_t = `inv`: slot 1 on line 3 whatever frame 3 shows
    lines = episode(
        world(), inv(), inv(press=[0], at=S1), inv(hold=[0], at=S1), world()
    )
    expected = GuiActions([S("left_click_1", 2, 2)], fs(), fs())
    assert run(lines, {2: "inv", 3: cls}) == expected


def test_hold_ends_with_the_span():
    lines = episode(world(), inv(), inv(press=[0], at=S1), world(hold=[0], at=S2),
                    world(hold=[0], at=OUT), world())  # fmt: skip
    assert run(lines, {2: "inv"}) == GuiActions([S("left_click_1", 2, 2)], fs(), fs())


def test_presses_inside_one_run_of_b():
    # B ∋ 0 on lines 2-5, pressed again on 4: holds 2..3 on slot 1, 4..5 on slot 2
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(hold=[0], at=S1),
                    inv(press=[0], at=S2), inv(hold=[0], at=S2), inv(), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("left_click_2", 4, 4)]
    assert run(lines, {2: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs())


def test_hold_ends_only_at_a_press_of_its_button():
    # the right press on line 3 does not end the left hold 2..3, which crosses slots
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[1], hold=[0], at=S2),
                    inv(), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("right_click_2", 3, 3)]
    assert run(lines, {2: "inv", 3: "inv"}) == GuiActions(segs, fs(), fs({2, 3}))


def test_press_outside_has_no_hold():
    lines = episode(world(), inv(), inv(press=[0], at=OUT), inv(hold=[0], at=S1),
                    inv(hold=[0], at=S2), world())  # fmt: skip
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs({2}))
