"""Spans, first frames, (1) needed frames, (2) cut and the first-line rule of
specs/vpt-gui-actions.md; hand cases, tiny catalogue."""

import dataclasses

import pytest
from gui_action_lines import MS, S1, S2, S10, SCREENS, box, episode, inv, world

from mcpref.gui.actions import (
    GuiActions,
    excluded_states,
    gui_actions,
    needed_frames,
    spans,
)
from mcpref.vpt.actions import Segment as S

fs = frozenset


def run(lines: list[dict], classes: dict) -> GuiActions:
    return gui_actions(lines, classes, SCREENS, MS)


def three_spans() -> list[dict]:
    """GUI from line 0, a 1-line span, a span up to the episode's end."""
    return episode(inv(), inv(), world(), box(), world(), world(), box(), inv(), box())


def test_spans_in_line_order():
    assert spans(three_spans()) == [(0, 1), (3, 3), (6, 8)]
    assert spans(episode(world(), inv(), box(), world())) == [(1, 2)]


def test_no_span_without_a_gui():
    assert spans(episode(world(), world())) == []


def test_needed_frames_are_the_first_frames():
    # u = k + 1 for [0, 1] and [6, 8], u = k for [3, 3]
    assert needed_frames(three_spans()) == [1, 3, 7]


def test_needed_frames_add_press_and_digit_lines():
    lines = episode(
        world(press=[0], keys=["2"]),  # 0: not in a span
        inv(),  # 1: span [1, 7], u = 2
        inv(press=[0]),  # 2: u and a press, listed once
        inv(press=[2]),  # 3: middle button only
        inv(keys=["q", "e", "left.shift", "0"]),  # 4: no digit of D
        inv(press=[1]),  # 5: right press
        inv(keys=["9"]),  # 6: digit pressed
        inv(keys=["9"]),  # 7: digit held, not in newKeys
        world(),
        inv(press=[0, 1]),  # 9: span [9, 10], cut or not
        inv(),  # 10: u
    )
    assert needed_frames(lines) == [2, 5, 6, 9, 10]


def test_needed_frames_include_an_auto_repeated_digit():
    # (1): N_t ∩ D ≠ ∅ holds for an OS auto-repeat too
    lines = episode(world(), inv(), inv(keys=["4"]), inv(keys=["4"], again=["4"]),
                    inv(keys=["4"]), world())  # fmt: skip
    assert needed_frames(lines) == [2, 3]


def test_needed_frames_include_the_first_line_of_a_container():
    # the first-line rule covers (3)-(8), not (1)
    lines = episode(world(), box(press=[0]), box(), box(keys=["5"]), world())
    assert needed_frames(lines) == [1, 2, 3]


@pytest.mark.parametrize("cls", ["other", "pause", "death"])
def test_span_cut_when_its_first_frame_is_other_pause_or_death(cls):
    # span [1, 3], u = 2: a press outside, Q, a press on a slot, a digit, a close
    lines = episode(world(), inv(press=[0], keys=["q"]), inv(press=[0], at=S1),
                    inv(keys=["2", "escape"], at=S10), world())  # fmt: skip
    assert run(lines, {1: "inv", 2: cls, 3: "inv"}) == GuiActions(
        [], fs({1, 2, 3}), fs()
    )


def test_span_opened_without_an_action_cut_on_death():
    # revision of 2026-10-21: the death screen's span is cut, its presses and close too
    lines = episode(
        world(), box(), box(press=[0], at=S1), box(keys=["escape"]), world()
    )
    assert run(lines, {2: "death"}) == GuiActions([], fs({1, 2, 3}), fs())


def test_cut_reads_frame_k_plus_1():
    lines = episode(world(), inv(press=[0], at=S1), inv(), inv(), world())
    # kept: the press on line 1 sees an `other` frame
    assert run(lines, {1: "other", 2: "inv"}) == GuiActions([], fs(), fs({1}))
    assert run(lines, {1: "inv", 2: "other"}) == GuiActions([], fs({1, 2, 3}), fs())


@pytest.mark.parametrize(
    "cls, expected",
    [
        ("inv", GuiActions([S("left_click_1", 1, 1)], fs(), fs())),
        ("other", GuiActions([], fs({1}), fs())),
    ],
)
def test_one_line_span_reads_its_own_frame(cls, expected):
    lines = episode(world(), inv(press=[0], at=S1), world())
    assert run(lines, {1: cls}) == expected


def test_episode_starting_in_a_container():
    # k = 0 is outside the first-line rule; u = 1
    lines = episode(box(press=[0], at=S1), box(), world())
    expected = GuiActions([S("left_click_1", 0, 0)], fs(), fs())
    assert run(lines, {0: "box", 1: "box"}) == expected
    assert run(lines, {0: "box", 1: "other"}) == GuiActions([], fs({0, 1}), fs())


@pytest.mark.parametrize("cls", ["box", "other"])
def test_first_line_of_a_container_is_ignored(cls):
    # line 1: left and right press, a digit over slot 10, F, then a drag to slot 2
    lines = episode(world(), box(press=[0, 1], at=S10, keys=["2", "f"]),
                    box(hold=[0], at=S2), box(), world())  # fmt: skip
    assert run(lines, {1: cls, 2: "box"}) == GuiActions([], fs(), fs())


def test_one_line_container_span_has_no_close():
    lines = episode(world(), box(keys=["escape"]), world())
    assert run(lines, {1: "box"}) == GuiActions([], fs(), fs())


def test_first_line_of_the_player_inventory_is_mapped():
    lines = episode(
        world(keys=["e"]), inv(press=[1], at=S1, keys=["q"]), inv(), world()
    )
    expected = GuiActions([S("right_click_1", 1, 1)], fs(), fs({1}))
    assert run(lines, {1: "inv", 2: "inv"}) == expected


def scene() -> tuple[list[dict], dict]:
    """A kept span [1, 3] (press 1, u 2, digit 3) and a cut span [5, 6] (press 5, u 6)."""
    lines = episode(world(), inv(press=[0], at=S1), inv(), inv(keys=["2"], at=S10),
                    world(), inv(press=[1], at=S1), inv(), world())  # fmt: skip
    return lines, {1: "inv", 2: "inv", 3: "inv", 5: "inv", 6: "other"}


def test_scene_with_every_needed_frame():
    lines, classes = scene()
    chain = [
        S("left_click_10", 3, 3),
        S("left_click_2", 3, 3),
        S("left_click_10", 3, 3),
    ]
    expected = GuiActions([S("left_click_1", 1, 1), *chain], fs({5, 6}), fs())
    assert run(lines, classes) == expected


@pytest.mark.parametrize("missing", [1, 2, 3, 5, 6])
def test_missing_needed_frame_raises_key_error(missing):
    lines, classes = scene()
    del classes[missing]
    with pytest.raises(KeyError):
        run(lines, classes)


def test_gui_actions_record():
    names = [f.name for f in dataclasses.fields(GuiActions)]
    assert names == ["segments", "cut", "foreign"]
    res = run(*scene())
    assert type(res.segments) is list
    assert type(res.cut) is frozenset and type(res.foreign) is frozenset
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.cut = fs()


def test_no_output(capfd):
    lines, classes = scene()
    run(lines, classes)
    excluded_states(lines, classes, {5, 6})
    assert capfd.readouterr() == ("", "")
