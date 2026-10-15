"""Turns (7) and the actions of an interval (8) of specs/vpt-actions.md, hand cases."""

import pytest
from action_lines import episode, repeat

from mcpref.vpt.actions import Segment, interval_actions, segments

S = Segment
RIGHT, LEFT, UP, DOWN = "turn_right", "turn_left", "turn_up", "turn_down"


def flat(n: int, yaw=None, pitch=None) -> list[dict]:
    return episode(repeat("-", n), yaw=yaw, pitch=pitch)


def test_segment_spanning_intervals_counts_in_each():
    segs, lines = [S("forward", 2, 9)], flat(13)
    found = [interval_actions(segs, lines, a, a + 4, (), 10.0) for a in (0, 4, 8)]
    assert found == [["forward"]] * 3


def test_segments_with_a_line_in_the_interval():
    segs = [S("attack", 0, 3), S("use", 4, 4), S("sneak", 7, 7), S("forward", 8, 9),
            S("back", 1, 10), S("left", 3, 4)]  # fmt: skip
    assert interval_actions(segs, flat(12), 4, 8, (), 10.0) == [
        "use",
        "sneak",
        "back",
        "left",
    ]


def test_cut_lines_are_not_in_the_interval():
    segs = [S("attack", 4, 5), S("use", 5, 7), S("sneak", 6, 6), S("back", 1, 3)]
    lines = flat(10)
    assert interval_actions(segs, lines, 4, 8, {4, 5, 6}, 10.0) == ["use"]
    assert interval_actions(segs, lines, 4, 8, [0, 2, 8, 9], 10.0) == [
        "attack",
        "use",
        "sneak",
    ]
    assert interval_actions(segs, lines, 4, 8, range(4, 8), 10.0) == []


def test_actions_in_the_order_of_segs_then_yaw_then_pitch():
    segs = [S("use", 6, 6), S("forward", 4, 5), S("hotbar_3", 5, 5), S("attack", 4, 5),
            S("attack", 6, 7)]  # fmt: skip
    lines = flat(9, yaw=[0, 0, 0, 0, 0, 0, -10, -20, -20], pitch=[0] * 8 + [10])
    found = interval_actions(segs, lines, 4, 8, (), 10.0)
    assert found == ["use", "forward", "hotbar_3", "attack", "attack", LEFT, LEFT, DOWN]


# (7): wrap(x) = x - 360 ceil((x - 180) / 360); n = floor(|sum| / rho + 1/2).
YAW = [
    ([0, 5], [RIGHT]),  # exactly half a turn
    ([0, 4.5], []),
    ([0, -5], [LEFT]),
    ([0, 14.5], [RIGHT]),
    ([0, 15], [RIGHT] * 2),
    ([0, 2.5, 5], [RIGHT]),
    ([2962.5, 82.5], []),  # a 2880 degree jump: wrap(-2880) = 0
    ([2962.5, 82.5, 87.5], [RIGHT]),
    ([-1701.5, -261.5, -266.5], [LEFT]),
    ([0, 170], [RIGHT] * 17),  # a real flick
    ([0, 170, 340], [RIGHT] * 34),  # two flicks: 340, not wrap(340) = -20
    ([0, -170, -340], [LEFT] * 34),
    ([0, 190], [LEFT] * 17),
    ([0, 180], [RIGHT] * 18),  # wrap(180) = 180
    ([0, -180], [RIGHT] * 18),  # wrap(-180) = 180
    ([0, 365], [RIGHT]),
    ([10, 5, 10], []),
    ([350, 10], [RIGHT] * 2),
]


@pytest.mark.parametrize("yaw, turns", YAW)
def test_yaw_turns(yaw, turns):
    lines = flat(len(yaw), yaw=yaw)
    assert interval_actions([], lines, 0, len(yaw) - 1, (), 10.0) == turns


PITCH = [
    ([0, 30], [DOWN] * 3),
    ([0, -30], [UP] * 3),
    ([0, 5], [DOWN]),  # exactly half a turn
    ([0, -5], [UP]),
    ([0, 4.5], []),
    ([0, 50, 0], []),  # only the end states count
    ([-90, 90], [DOWN] * 18),
    ([20, 0, 14.5], [UP]),
]


@pytest.mark.parametrize("pitch, turns", PITCH)
def test_pitch_turns(pitch, turns):
    lines = flat(len(pitch), pitch=pitch)
    assert interval_actions([], lines, 0, len(pitch) - 1, (), 10.0) == turns


@pytest.mark.parametrize("turn_deg, n", [(2.5, 3), (4.0, 2), (12.5, 1), (12.6, 0)])
def test_turn_deg(turn_deg, n):
    # |6.25| / turn_deg + 1/2: 3.0, 2.06, 1.0, 0.996.
    lines = flat(2, yaw=[0, 6.25], pitch=[0, -6.25])
    assert interval_actions([], lines, 0, 1, (), turn_deg) == [RIGHT] * n + [UP] * n


def test_turns_only_between_the_two_states():
    lines = flat(6, yaw=[0, 90, 90, 100, 100, 300], pitch=[-40, 0, 30, 0, 10, 80])
    assert interval_actions([], lines, 1, 4, (), 10.0) == [RIGHT, DOWN]


def test_turns_count_cut_lines():
    # (7) sums wrap over k = a..b-1, whatever the consumer cuts.
    lines = flat(4, yaw=[0, 20, 20, 20], pitch=[0, 0, 30, 30])
    assert interval_actions([], lines, 0, 3, {0, 1}, 10.0) == [RIGHT] * 2 + [DOWN] * 3


def test_with_segments_of_an_episode():
    lines = episode("- - w w w+M0 w w w w w - - -", yaw=[0] * 12 + [-20])
    segs = segments(lines, r_min=3)
    assert segs == [S("forward", 2, 9), S("attack", 4, 4)]
    found = [interval_actions(segs, lines, a, a + 4, (), 10.0) for a in (0, 4, 8)]
    assert found == [["forward"], ["forward", "attack"], ["forward", LEFT, LEFT]]
