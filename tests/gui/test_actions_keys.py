"""(5) close, (6) digits, (7) foreign keys and the order (9) of specs/vpt-gui-actions.md;
hand cases, tiny catalogue."""

import pytest
from gui_action_lines import (
    B37,
    C37,
    MS,
    OUT,
    S1,
    S2,
    S10,
    SCREENS,
    episode,
    inv,
    world,
)

from mcpref.gui.actions import GuiActions, gui_actions
from mcpref.vpt.actions import Segment as S

fs = frozenset


def run(lines: list[dict], classes: dict) -> GuiActions:
    return gui_actions(lines, classes, SCREENS, MS)


def chain(hovered: int, s: int, t: int) -> list[S]:
    """(6): the clicks (hovered, s, hovered) on line t."""
    return [S(f"left_click_{i}", t, t) for i in (hovered, s, hovered)]


@pytest.mark.parametrize("key", ["e", "escape"])
def test_close_on_the_last_line(key):
    # span [1, 3], j = 3 = T - 2
    lines = episode(world(), inv(), inv(), inv(keys=[key]), world())
    assert run(lines, {2: "inv"}) == GuiActions([S("close", 3, 3)], fs(), fs())


@pytest.mark.parametrize("key", ["e", "escape"])
def test_no_close_before_the_last_line(key):
    lines = episode(world(), inv(), inv(keys=[key]), inv(), world())
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


@pytest.mark.parametrize("key", ["e", "escape"])
def test_no_close_at_the_episode_end(key):
    lines = episode(world(), inv(), inv(), inv(keys=[key]))
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


def test_close_needs_a_new_key():
    lines = episode(world(), inv(), inv(keys=["e"]), inv(keys=["e"]), world())
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


@pytest.mark.parametrize(
    "digit, at, cls, segs",
    [
        ("2", S10, "inv", chain(10, 2, 3)),  # over a slot
        ("1", S2, "inv", chain(2, 1, 3)),
        ("2", C37, "box", chain(37, 2, 3)),
        ("2", S2, "inv", []),  # over slot s
        ("2", B37, "inv", []),  # over a button
        ("2", OUT, "inv", []),  # over nothing
        ("3", C37, "box", []),  # `box` has no element 3
        ("9", S10, "inv", []),
        ("2", S1, "lectern", []),  # a button where `inv` has slot 1
        ("2", S10, "other", []),
        ("2", S10, "pause", []),
    ],
)
def test_digit(digit, at, cls, segs):
    lines = episode(world(), inv(), inv(), inv(keys=[digit], at=at), world())
    assert run(lines, {2: "inv", 3: cls}) == GuiActions(segs, fs(), fs())


def test_held_digit_swaps_once():
    # line 3 holds 2 without a press, line 4 auto-repeats it
    lines = episode(world(), inv(), inv(keys=["2"], at=S10), inv(keys=["2"], at=S10),
                    inv(keys=["2"], again=["2"], at=S10), world())  # fmt: skip
    assert run(lines, {2: "inv", 4: "inv"}) == GuiActions(chain(10, 2, 2), fs(), fs())


def test_digit_pressed_again_swaps_again():
    lines = episode(world(), inv(), inv(keys=["2"], at=S10), inv(at=S10),
                    inv(keys=["2"], at=S10), world())  # fmt: skip
    segs = chain(10, 2, 2) + chain(10, 2, 4)
    assert run(lines, {2: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs())


def test_digit_on_line_0():
    # t = 0 needs no line before it (the last line holds 2)
    lines = episode(inv(keys=["2"], at=S10), inv(), world(keys=["2"]))
    assert run(lines, {0: "inv", 1: "inv"}) == GuiActions(chain(10, 2, 0), fs(), fs())


def test_digit_held_before_the_span_swaps_nothing():
    lines = episode(world(keys=["2"]), inv(keys=["2"], again=["2"], at=S10), inv(),
                    world())  # fmt: skip
    assert run(lines, {1: "inv", 2: "inv"}) == GuiActions([], fs(), fs())


@pytest.mark.parametrize("key", ["q", "f"])
def test_q_and_f_make_their_line_foreign(key):
    lines = episode(world(), inv(), inv(), inv(keys=[key]), inv(), world())
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs({3}))


def test_other_keys_are_ignored():
    keys = ["enter", "w", "space", "left.control", "e", "t", "f1", "f3", "f5", "0"]
    lines = episode(world(), inv(), inv(), inv(keys=keys, at=S10), inv(), world())
    assert run(lines, {2: "inv"}) == GuiActions([], fs(), fs())


def test_order_within_a_line():
    # (3) in the order of newButtons, then (5), then (6) in the order of newKeys
    lines = episode(world(), inv(), inv(),
                    inv(press=[1, 0], at=S10, keys=["3", "escape", "2"]), world())  # fmt: skip
    segs = [S("right_click_10", 3, 3), S("left_click_10", 3, 3), S("close", 3, 3),
            *chain(10, 3, 3), *chain(10, 2, 3)]  # fmt: skip
    assert run(lines, {2: "inv", 3: "inv"}) == GuiActions(segs, fs(), fs())


@pytest.mark.parametrize(
    "first, segs",
    [
        ([1, 0], [S("right_click_1", 2, 2), S("double_click_1", 2, 4)]),
        ([0, 1], [S("double_click_1", 2, 4), S("right_click_1", 2, 2)]),
    ],
)
def test_pair_at_the_place_of_its_first_press(first, segs):
    lines = episode(world(), inv(), inv(press=first, at=S1), inv(press=[1], at=S2),
                    inv(press=[0], at=S1), world())  # fmt: skip
    expected = GuiActions([*segs, S("right_click_2", 3, 3)], fs(), fs())
    assert run(lines, {2: "inv", 3: "inv", 4: "inv"}) == expected


def test_segments_by_first_line():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(keys=["2"], at=S10),
                    inv(press=[0], at=S1), inv(press=[1], at=S2, keys=["escape"]),
                    world())  # fmt: skip
    segs = [S("double_click_1", 2, 4), *chain(10, 2, 3), S("right_click_2", 5, 5),
            S("close", 5, 5)]  # fmt: skip
    found = run(lines, {2: "inv", 3: "inv", 4: "inv", 5: "inv"})
    assert found == GuiActions(segs, fs(), fs())
