"""(3) of specs/vpt-gui-actions.md, revision of 2026-10-21: a right press, with or without shift,
on a button for which `right_handled` is false is ignored (no segment, not foreign, no hold of
its own, though it ends an earlier hold of the right button); hand cases, tiny catalogue."""

import pytest
from gui_action_lines import (
    B37,
    C37,
    MS,
    OUT,
    S1,
    S2,
    SCREENS,
    episode,
    inv,
    screen,
    slot,
    world,
)

from mcpref.gui.actions import GuiActions, gui_actions
from mcpref.gui.screens import Element, Rect
from mcpref.vpt.actions import Segment as S
from mcpref.vpt.actions import interval_actions

fs = frozenset
SHIFTS = [[], ["left.shift"], ["right.shift"]]


def named(index: int, name: str, x: int, y: int, w: int, h: int) -> Element:
    return Element(
        index, name, "button", Rect(name, x, y, w, h), (x + w / 2, y + h / 2)
    )


# `cutter`: slot 1 as in `inv`; under B37 the button 37 `recipe_book_button` (not
# right-handled), under C37 the button 38 `recipe_0` (right-handled).
CUTTER = screen("cutter", slot(1, 100, 100), named(37, "recipe_book_button", 160, 100, 20, 13),
                named(38, "recipe_0", 100, 50, 18, 18))  # fmt: skip
CATALOGUE = {**SCREENS, "cutter": CUTTER}


def run(lines: list[dict], classes: dict) -> GuiActions:
    return gui_actions(lines, classes, CATALOGUE, MS)


def one_press(keys: list, press: list, at: tuple, cls: str) -> GuiActions:
    lines = episode(world(), inv(), inv(keys=keys, press=press, at=at), inv(), world())
    return run(lines, {2: cls})


@pytest.mark.parametrize("keys", SHIFTS)
def test_right_press_on_a_right_handled_button_is_kept(keys):
    action = "shift_click_38" if keys else "right_click_38"
    found = one_press(keys, [1], C37, "cutter")
    assert found == GuiActions([S(action, 2, 2)], fs(), fs())


@pytest.mark.parametrize("keys", SHIFTS)
@pytest.mark.parametrize(
    "cls, at", [("cutter", B37), ("inv", B37), ("lectern", S1), ("lectern", S2)]
)
def test_right_press_on_another_button_is_ignored(keys, cls, at):
    # recipe_book_button of `cutter`, b37 of `inv`, b37 and b38 of `lectern`
    assert one_press(keys, [1], at, cls) == GuiActions([], fs(), fs())


@pytest.mark.parametrize("keys", SHIFTS)
def test_left_press_on_such_a_button_is_kept(keys):
    action = "shift_click_37" if keys else "left_click_37"
    found = one_press(keys, [0], B37, "cutter")
    assert found == GuiActions([S(action, 2, 2)], fs(), fs())


@pytest.mark.parametrize("press", [[1, 0], [0, 1]])
def test_left_press_on_the_same_line_is_kept(press):
    found = one_press([], press, B37, "inv")
    assert found == GuiActions([S("left_click_37", 2, 2)], fs(), fs())


@pytest.mark.parametrize("keys", SHIFTS)
def test_ignored_press_has_no_hold(keys):
    # the cursor leaves the button while it is held: nothing is foreign
    lines = episode(world(), inv(), inv(keys=keys, press=[1], at=B37),
                    inv(keys=keys, hold=[1], at=S1), inv(hold=[1], at=OUT), inv(),
                    world())  # fmt: skip
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


def test_kept_right_press_on_a_button_has_its_hold():
    lines = episode(world(), inv(), inv(press=[1], at=C37), inv(hold=[1], at=S1),
                    inv(hold=[1], at=OUT), inv(), world())  # fmt: skip
    expected = GuiActions([S("right_click_38", 2, 2)], fs(), fs({2, 3, 4}))
    assert run(lines, {2: "cutter"}) == expected


@pytest.mark.parametrize("keys", SHIFTS)
def test_ignored_press_ends_an_earlier_hold(keys):
    # B ∋ 1 on lines 2-5: the hold of the press on slot 1 is 2..3; line 4 presses b37 (ignored)
    lines = episode(world(), inv(), inv(press=[1], at=S1), inv(hold=[1], at=S1),
                    inv(keys=keys, press=[1], at=B37), inv(hold=[1], at=S2), inv(),
                    world())  # fmt: skip
    expected = GuiActions([S("right_click_1", 2, 2)], fs(), fs())
    assert run(lines, {2: "inv", 4: "inv"}) == expected


def test_without_the_second_press_the_hold_crosses_elements():
    # the contrast of the test above: one hold 2..5 leaving slot 1
    lines = episode(world(), inv(), inv(press=[1], at=S1), inv(hold=[1], at=S1),
                    inv(hold=[1], at=B37), inv(hold=[1], at=S2), inv(),
                    world())  # fmt: skip
    expected = GuiActions([S("right_click_1", 2, 2)], fs(), fs({2, 3, 4, 5}))
    assert run(lines, {2: "inv"}) == expected


def test_interval_n_without_the_ignored_press():
    # lines 2, 3, 4: left on slot 1, right on b37 (ignored), shift + right on b37 (ignored)
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[1], at=B37),
                    inv(keys=["left.shift"], press=[1], at=B37), inv(press=[1], at=S2),
                    inv(), world())  # fmt: skip
    found = run(lines, {2: "inv", 3: "inv", 4: "inv", 5: "inv"})
    segs = [S("left_click_1", 2, 2), S("right_click_2", 5, 5)]
    assert found == GuiActions(segs, fs(), fs())
    n = [len(interval_actions(found.segments, lines, a, a + 2, found.cut, 10.0))
         for a in (1, 3, 5)]  # fmt: skip
    assert n == [1, 0, 1]
