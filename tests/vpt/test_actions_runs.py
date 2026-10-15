"""Effects (2), smoothing (3) and splits (4) of specs/vpt-actions.md, hand cases."""

import pytest
from action_lines import CUSTOM, bump, episode, repeat

from mcpref.vpt.actions import Segment, segments

S = Segment
F = "forward"
JF = "jump_forward"
MINE, USE = (
    "minecraft.mine_block:minecraft.stone",
    "minecraft.use_item:minecraft.oak_planks",
)
JUMP = CUSTOM + "jump"
ATTACK_STATS = ["minecraft.mine_block:minecraft.stone", "minecraft.mine_block:minecraft.oak_log",
                CUSTOM + "damage_dealt", CUSTOM + "damage_dealt_absorbed",
                CUSTOM + "damage_dealt_resisted"]  # fmt: skip
OTHER_STATS = [CUSTOM + "damage_taken", CUSTOM + "mob_kills", "minecraft.broken:minecraft.stone_pickaxe",
               "minecraft.killed:minecraft.zombie", CUSTOM + "walk_one_cm"]  # fmt: skip


def found(tokens: str, r_min: int = 3) -> list:
    return segments(episode(tokens), r_min=r_min)


SMOOTHING = [  # (tokens, r_min, expected); r_min = 3 unless stated
    # (3a): a release shorter than r_min inside a hold takes the hold's value.
    ("w w w - w w w", 3, [S(F, 0, 6)]),
    ("w w w - - w w w", 3, [S(F, 0, 7)]),
    ("w w w - - - w w w", 3, [S(F, 0, 2), S(F, 6, 8)]),
    ("w+space w+space - w+space w+space", 3, [S(JF, 0, 4)]),
    ("w w w - d d d", 3, [S(F, 0, 2), S("right", 4, 6)]),
    ("- w w w -", 3, [S(F, 1, 3)]),
    # (3a): a GUI line is no neighbour and no part of a release.
    ("w w w - # - w w w", 4, [S(F, 0, 2), S(F, 6, 8)]),
    ("w w w # w w w", 3, [S(F, 0, 2), S(F, 4, 6)]),
    # (3b): a j = 0 run shorter than r_min between two values takes the value before it.
    ("w w w w+d d d d", 3, [S(F, 0, 3), S("right", 4, 6)]),
    ("w w w w+d w+d d d d", 3, [S(F, 0, 4), S("right", 5, 7)]),
    (
        "w w w w+d w+d w+d d d d",
        3,
        [S(F, 0, 2), S("forward_right", 3, 5), S("right", 6, 8)],
    ),
    ("w+space w+space w d d", 3, [S(JF, 0, 2), S("right", 3, 4)]),
    (
        "w w w w+d d a a a",
        3,
        [S(F, 0, 4), S("left", 5, 7)],
    ),  # in line order, as smoothed
    (
        "d w w w+space w+space",
        3,
        [S("right", 0, 2), S(JF, 3, 4)],
    ),  # right: no line before
    # (3b): isolated short taps and short jump runs are kept.
    ("- - w - -", 3, [S(F, 2, 2)]),
    ("w w w w+space d d d", 3, [S(F, 0, 2), S(JF, 3, 3), S("right", 4, 6)]),
    ("w w w w+d - d d d", 3, [S(F, 0, 2), S("forward_right", 3, 3), S("right", 5, 7)]),
    ("w w w w+d # d d d", 3, [S(F, 0, 2), S("forward_right", 3, 3), S("right", 5, 7)]),
    # (3a) before (3b): the filled run d - d is shorter than 4 and takes forward.
    ("w w w d - d w w w", 4, [S(F, 0, 8)]),
    ("w w w d - d w w w", 3, [S(F, 0, 2), S("right", 3, 5), S(F, 6, 8)]),
    # r_min = 1: nothing is shorter.
    (
        "w - w w+d d",
        1,
        [S(F, 0, 0), S(F, 2, 2), S("forward_right", 3, 3), S("right", 4, 4)],
    ),
]


@pytest.mark.parametrize("tokens, r_min, expected", SMOOTHING)
def test_smoothing(tokens, r_min, expected):
    assert found(tokens, r_min) == expected


@pytest.mark.parametrize("stat", ATTACK_STATS)
def test_attack_split_by_mining_and_damage(stat):
    lines = bump(episode(repeat("M0", 10)), stat, 4)
    assert segments(lines, r_min=3) == [S("attack", 0, 4), S("attack", 5, 9)]


@pytest.mark.parametrize("stat", [USE, JUMP, *OTHER_STATS])
def test_attack_not_split_by_other_stats(stat):
    assert segments(bump(episode(repeat("M0", 10)), stat, 4), r_min=3) == [
        S("attack", 0, 9)
    ]


@pytest.mark.parametrize("stat", [USE, "minecraft.use_item:minecraft.bread"])
def test_use_split_by_use_item(stat):
    lines = bump(episode(repeat("M1", 10)), stat, 4)
    assert segments(lines, r_min=3) == [S("use", 0, 4), S("use", 5, 9)]


@pytest.mark.parametrize("stat", [MINE, CUSTOM + "damage_dealt", JUMP, *OTHER_STATS])
def test_use_not_split_by_other_stats(stat):
    assert segments(bump(episode(repeat("M1", 10)), stat, 4), r_min=3) == [
        S("use", 0, 9)
    ]


@pytest.mark.parametrize("token, action", [("w+space", JF), ("space", "jump"),
                         ("s+a+space", "jump_back_left")])  # fmt: skip
def test_jump_splits_move_with_space_held(token, action):
    lines = bump(episode(repeat(token, 10)), JUMP, 4)
    assert segments(lines, r_min=3) == [S(action, 0, 4), S(action, 5, 9)]


def test_no_jump_split_without_space_held():
    assert segments(bump(episode(repeat("w", 10)), JUMP, 4), r_min=3) == [S(F, 0, 9)]
    # Line 4 holds no space (j_4 = 0 by (1)); (3a) gives it the value jump_forward.
    lines = bump(episode(f"{repeat('w+space', 4)} - {repeat('w+space', 5)}"), JUMP, 4)
    assert segments(lines, r_min=3) == [S(JF, 0, 9)]


def test_jump_split_inside_a_smoothed_run():
    lines = bump(episode(f"{repeat('w+space', 4)} - {repeat('w+space', 5)}"), JUMP, 6)
    assert segments(lines, r_min=3) == [S(JF, 0, 6), S(JF, 7, 9)]


@pytest.mark.parametrize("stat", [MINE, USE, JUMP, CUSTOM + "damage_dealt"])
def test_move_and_sneak_not_split_by_other_channels(stat):
    lines = bump(episode(repeat("w+left.shift", 10)), stat, 4)
    assert segments(lines, r_min=3) == [S(F, 0, 9), S("sneak", 0, 9)]


def test_no_split_on_the_last_line_or_outside_a_run():
    lines = bump(episode(f"{repeat('M0', 5)} {repeat('-', 5)}"), MINE, 4, 6, 8)
    assert segments(lines, r_min=1) == [S("attack", 0, 4)]
    gui = bump(episode(f"{repeat('M0', 5)} # M0 M0 M0"), MINE, 4, 5)
    assert segments(gui, r_min=1) == [S("attack", 0, 4), S("attack", 6, 8)]


@pytest.mark.parametrize("after, pieces", [
    ((0, 1, 4, 7, 8), [(0, 4), (5, 9)]),  # short first and last pieces merged
    ((3, 4), [(0, 3), (4, 4), (5, 9)]),  # a short middle piece kept
    ((2, 6), [(0, 2), (3, 6), (7, 9)]),  # first and last pieces of exactly r_min kept
    ((1, 2, 3, 4, 5, 6, 7), [(0, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 9)]),
    ((0, 1, 7, 8), [(0, 9)]),
])  # fmt: skip
def test_split_pieces(after, pieces):
    lines = bump(episode(repeat("M0", 10)), MINE, *after)
    assert segments(lines, r_min=3) == [S("attack", p, q) for p, q in pieces]


def test_short_run_with_splits_stays_whole():
    lines = bump(episode("M0 M0 M0 M0 -"), CUSTOM + "damage_dealt", 0, 1, 2)
    assert segments(lines, r_min=3) == [S("attack", 0, 3)]


def test_two_stats_on_one_line_split_once():
    lines = bump(bump(episode(repeat("M0", 10)), MINE, 4), CUSTOM + "damage_dealt", 4)
    assert segments(lines, r_min=3) == [S("attack", 0, 4), S("attack", 5, 9)]
