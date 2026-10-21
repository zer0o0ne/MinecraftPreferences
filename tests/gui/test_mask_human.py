"""(6) of specs/action-mask.md: after the revision of block 21 the human data holds no forbidden
action. Hand-made lines on the approved catalogue through `gui_actions` (every element of every
class of D pressed left, right, shift + left, shift + right, double-clicked if a slot, a digit,
a close) and the world segments and turns of specs/vpt-actions.md; each action allowed by the
mask of the line where it is pressed; the right presses on other buttons absent."""

import pytest
from gui_action_lines import MS, episode, inv, px, world
from mask_oracle import D, names, right_handled

from mcpref.gui.actions import GuiActions, gui_actions
from mcpref.gui.mask import state_mask
from mcpref.vpt.actions import Segment as S
from mcpref.vpt.actions import interval_actions, segments

A = names(90)
SHIFT = ["left.shift"]


def pressed(elements: list[dict]) -> tuple[list[dict], list[S]]:
    """Rows of one span and its expected segments by (3), (3a), (5), (6) of block 21; presses
    1 s apart, each followed by a release line; a double click's presses 100 ms apart."""
    rows, expected = [world(), inv()], []

    def add(action: str | None, last: int = 0, **kw) -> None:
        t = len(rows)
        if action:
            expected.append(S(action, t, t + last))
        rows.append(inv(ms=1000 * t, **kw))
        rows.append(inv(ms=1000 * t + 50))

    for e in elements:
        i, at, slot = e["index"], px(*e["click"]), e["kind"] == "slot"
        right = slot or right_handled(e)
        add(f"left_click_{i}", press=[0], at=at)
        add(f"right_click_{i}" if right else None, press=[1], at=at)
        add(f"shift_click_{i}", press=[0], at=at, keys=SHIFT)
        add(f"shift_click_{i}" if right else None, press=[1], at=at, keys=SHIFT)
        if slot:
            add(f"double_click_{i}", last=2, press=[0], at=at)
            rows.append(inv(press=[0], at=at, ms=1000 * (len(rows) - 2) + 100))
            rows.append(inv(ms=1000 * len(rows)))
    last = max(elements, key=lambda e: e["index"])
    t, i = len(rows), last["index"]
    if last["kind"] == "slot" and i != 1 and any(e["index"] == 1 for e in elements):
        chain = (f"left_click_{i}", "left_click_1", f"left_click_{i}")
        expected += [S(a, t, t) for a in chain]
    rows.append(inv(keys=["1"], at=px(*last["click"]), ms=1000 * t))
    expected.append(S("close", t + 1, t + 1))
    rows += [inv(keys=["escape"], ms=1000 * (t + 1)), world()]
    return rows, expected


@pytest.mark.parametrize("cls", D)
def test_gui_segments_are_allowed(masks, classes, approved_screens, cls):
    rows, expected = pressed(classes[cls]["elements"])
    lines = episode(*rows)
    found = gui_actions(lines, dict.fromkeys(range(1, len(lines) - 1), cls),
                        approved_screens, MS)  # fmt: skip
    assert found == GuiActions(expected, frozenset(), frozenset())
    for seg in found.segments:
        line = lines[seg.first]
        mask = state_mask(masks, True, cls, line["pitch"], line["hotbar"])
        assert mask[A.index(seg.action)], (cls, seg)


def world_lines() -> list[dict]:
    """Moves, attack, use, hotbar 0 -> 8 -> 3 -> 0, pitch from 90 up and from -90 down."""
    rows = [world(keys=["w"]), world(keys=["w"]), world(keys=["w"], press=[0]),
            world(hold=[0]), world(hold=[0]), world(press=[1]), world(keys=["e"]),
            world(keys=["space"]), world(keys=["space"]), world()]  # fmt: skip
    lines = episode(*rows)
    hotbar = [0, 0, 8, 8, 3, 3, 3, 0, 0, 0]
    pitch = [90.0, 90.0, 90.0, 80.0, 60.0, -90.0, -90.0, -90.0, -60.0, -60.0]
    for line, h, theta in zip(lines, hotbar, pitch, strict=True):
        line["hotbar"], line["pitch"] = h, theta
    return lines


def world_mask(masks, line: dict):
    return state_mask(masks, False, None, line["pitch"], line["hotbar"])


def test_world_segments_are_allowed(masks):
    lines = world_lines()
    segs = segments(lines, 3)
    hotbar = [S("hotbar_9", 1, 1), S("hotbar_4", 3, 3), S("hotbar_1", 6, 6)]
    assert [s for s in segs if s.action.startswith("hotbar_")] == hotbar
    for seg in segs:
        assert world_mask(masks, lines[seg.first])[A.index(seg.action)], seg


def test_turns_are_allowed_at_the_first_line_of_their_interval(masks):
    lines = world_lines()
    # (2, 4): 90 -> 60, 3 turn_up at 90; (4, 6): 60 -> -90, 15 turn_up; (6, 8): -90 -> -60,
    # 3 turn_down at -90
    turns = {a: interval_actions([], lines, a, a + 2, (), 10.0) for a in range(0, 8, 2)}
    up, down = ["turn_up"], ["turn_down"]
    assert turns == {0: [], 2: up * 3, 4: up * 15, 6: down * 3}
    for a, found in turns.items():
        mask = world_mask(masks, lines[a])
        assert all(mask[A.index(n)] for n in found), (a, found)
