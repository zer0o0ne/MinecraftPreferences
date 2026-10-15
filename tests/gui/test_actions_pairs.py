"""(3a) double clicks of specs/vpt-gui-actions.md, with (4) on their presses and (9)
through `interval_actions`; hand cases, tiny catalogue."""

import pytest
from gui_action_lines import B37, MS, OUT, S1, S2, SCREENS, episode, inv, world

from mcpref.gui.actions import GuiActions, gui_actions
from mcpref.vpt.actions import Segment as S
from mcpref.vpt.actions import interval_actions

fs = frozenset
PAIR = S("double_click_1", 2, 4)


def run(lines: list[dict], classes: dict) -> GuiActions:
    return gui_actions(lines, classes, SCREENS, MS)


def two_presses(first: tuple, second: tuple, ms=(100, 150, 200)) -> list[dict]:
    """Presses (button, position) on lines 2 and 4 of the span [1, 4], `milli` offsets."""
    return episode(world(), inv(), inv(press=[first[0]], at=first[1], ms=ms[0]),
                   inv(ms=ms[1]), inv(press=[second[0]], at=second[1], ms=ms[2]),
                   world(ms=ms[2] + 50))  # fmt: skip


@pytest.mark.parametrize("ms", [250, 300])
@pytest.mark.parametrize("below", [1, 0])
def test_pair_needs_a_gap_below_double_click_ms(ms, below):
    lines = two_presses((0, S1), (0, S1), (1000, 1010, 1000 + ms - below))
    clicks = [S("left_click_1", 2, 2), S("left_click_1", 4, 4)]
    found = gui_actions(lines, {2: "inv", 4: "inv"}, SCREENS, ms)
    assert found == GuiActions([PAIR] if below else clicks, fs(), fs())


@pytest.mark.parametrize(
    "first, second, actions",
    [
        ((0, S1), (0, S2), ["left_click_1", "left_click_2"]),  # another element
        ((0, B37), (0, B37), ["left_click_37", "left_click_37"]),  # not a slot
        ((1, S1), (1, S1), ["right_click_1", "right_click_1"]),
        ((0, S1), (1, S1), ["left_click_1", "right_click_1"]),
        ((1, S1), (0, S1), ["right_click_1", "left_click_1"]),
    ],
)
def test_no_pair(first, second, actions):
    segs = [S(actions[0], 2, 2), S(actions[1], 4, 4)]
    found = run(two_presses(first, second), {2: "inv", 4: "inv"})
    assert found == GuiActions(segs, fs(), fs())


def test_left_press_between_breaks_the_pair():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[0], at=S2),
                    inv(press=[0], at=S1), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("left_click_2", 3, 3), S("left_click_1", 4, 4)]
    assert run(lines, {2: "inv", 3: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs())


def test_foreign_left_press_between_breaks_the_pair():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[0], at=OUT),
                    inv(press=[0], at=S1), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("left_click_1", 4, 4)]
    assert run(lines, {2: "inv", 3: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs({3}))


def test_right_press_between_keeps_the_pair():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[1], at=S2),
                    inv(press=[0], at=S1), world())  # fmt: skip
    segs = [PAIR, S("right_click_2", 3, 3)]
    assert run(lines, {2: "inv", 3: "inv", 4: "inv"}) == GuiActions(segs, fs(), fs())


def presses_on_slot_1(n: int) -> list[dict]:
    """Left presses on slot 1 on lines 2, 4, ..., 100 ms apart, released between."""
    rows = [inv(press=[0], at=S1) if t % 2 == 0 else inv() for t in range(2, 2 * n + 1)]
    return episode(world(), inv(), *rows, world())


def test_triple_press_is_one_pair_and_one_click():
    found = run(presses_on_slot_1(3), {2: "inv", 4: "inv", 6: "inv"})
    assert found == GuiActions([PAIR, S("left_click_1", 6, 6)], fs(), fs())


def test_fourth_press_pairs_with_the_third():
    found = run(presses_on_slot_1(4), {2: "inv", 4: "inv", 6: "inv", 8: "inv"})
    assert found == GuiActions([PAIR, S("double_click_1", 6, 8)], fs(), fs())


def test_pair_inside_one_run_of_b():
    # released and pressed again within a record: B ∋ 0 on lines 2-4
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(press=[0], at=S1),
                    inv(hold=[0], at=S1), world())  # fmt: skip
    expected = GuiActions([S("double_click_1", 2, 3)], fs(), fs())
    assert run(lines, {2: "inv", 3: "inv"}) == expected


def test_no_pair_across_spans():
    # spans [1, 2] and [4, 5]; the presses on lines 2 and 4 are 100 ms apart
    lines = episode(world(), inv(), inv(press=[0], at=S1), world(at=S1),
                    inv(press=[0], at=S1), inv(), world())  # fmt: skip
    segs = [S("left_click_1", 2, 2), S("left_click_1", 4, 4)]
    assert run(lines, {2: "inv", 4: "inv", 5: "inv"}) == GuiActions(segs, fs(), fs())


def test_pair_across_two_intervals_counts_in_each():
    lines = episode(world(), inv(), inv(), inv(press=[0], at=S1), inv(),
                    inv(press=[0], at=S1), inv(), inv(), world())  # fmt: skip
    found = run(lines, {2: "inv", 3: "inv", 5: "inv"})
    assert found == GuiActions([S("double_click_1", 3, 5)], fs(), fs())
    for a in (0, 4):
        actions = interval_actions(found.segments, lines, a, a + 4, found.cut, 10.0)
        assert actions == ["double_click_1"]


@pytest.mark.parametrize(
    "first, second, foreign",
    [
        ([], [], False),
        (["left.shift"], [], False),  # only shift_t' counts
        ([], ["left.shift"], True),
        (["right.shift"], ["right.shift"], True),
    ],
)
def test_shift_double_click_is_foreign(first, second, foreign):
    lines = episode(world(), inv(), inv(keys=first, press=[0], at=S1), inv(keys=first),
                    inv(keys=second, press=[0], at=S1), world())  # fmt: skip
    found = run(lines, {2: "inv", 4: "inv"})
    shifted = GuiActions([], fs(), fs({2, 3, 4}))
    assert found == (shifted if foreign else GuiActions([PAIR], fs(), fs()))


def test_drag_on_the_second_press_of_a_pair():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(at=S1),
                    inv(press=[0], at=S1), inv(hold=[0], at=S2), inv(), world())  # fmt: skip
    assert run(lines, {2: "inv", 4: "inv"}) == GuiActions([PAIR], fs(), fs({4, 5}))


def test_drag_on_the_first_press_of_a_pair():
    lines = episode(world(), inv(), inv(press=[0], at=S1), inv(hold=[0], at=S2),
                    inv(press=[0], at=S1), world())  # fmt: skip
    assert run(lines, {2: "inv", 4: "inv"}) == GuiActions([PAIR], fs(), fs({2, 3}))


@pytest.mark.parametrize("cls", ["other", "pause", "lectern"])
def test_second_press_reads_its_own_frame(cls):
    # e_4 under class c_4: none, or button 37 of `lectern`, is not slot 1
    found = run(two_presses((0, S1), (0, S1)), {2: "inv", 4: cls})
    second = [S("left_click_37", 4, 4)] if cls == "lectern" else []
    foreign = fs() if cls == "lectern" else fs({4})
    assert found == GuiActions([S("left_click_1", 2, 2), *second], fs(), foreign)
