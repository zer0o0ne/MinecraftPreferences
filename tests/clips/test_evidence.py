"""Evidence and kappa of specs/clips.md (Notation): kept lines, S on the kept timeline merged
with the GUI segments by first line, F over all lines, Omega with the caller's start."""

import numpy as np
import pytest
from vpt.action_lines import CUSTOM, bump, episode

from mcpref.clips.evidence import Evidence, evidence, kept
from mcpref.vpt.actions import ActionsConfig
from mcpref.vpt.actions import Segment as S

from .clip_oracle import truth
from .clip_world import ACTIONS, OFF, RIDE, gui, random_episode

T_, F_ = True, False


def test_kept_lines_around_cut_spans():
    lines = episode("# # w w w # # # w w w w # #")
    ev = evidence(lines, gui(cut={0, 1, 5, 6, 7, 12, 13}), OFF, ACTIONS)
    assert kept(ev) == [2, 3, 4, 8, 9, 10, 11]


def test_every_line_is_kept_without_cut():
    lines = episode("# w w # @ w")
    assert kept(evidence(lines, gui(), OFF, ACTIONS)) == [0, 1, 2, 3, 4, 5]


def test_world_hold_across_a_cut_span_is_one_segment():
    lines = episode("w w w #:w #:w #:w #:w w w w")
    ev = evidence(lines, gui(cut=range(3, 7)), OFF, ACTIONS)
    assert ev.segments == [S("forward", 0, 9)]
    # the same span kept: its GUI lines end the hold
    ev = evidence(lines, gui(), OFF, ACTIONS)
    assert ev.segments == [S("forward", 0, 2), S("forward", 7, 9)]


def test_world_segments_are_smoothed_on_the_kept_timeline():
    # kept lines w w w - w w w: (3a) fills the one-line release (r_min = 3)
    lines = episode("w w w # # # - w w w")
    ev = evidence(lines, gui(cut={3, 4, 5}), OFF, ACTIONS)
    assert ev.segments == [S("forward", 0, 9)]
    # the span kept: line 6 follows a GUI line, nothing is filled
    ev = evidence(lines, gui(), OFF, ACTIONS)
    assert ev.segments == [S("forward", 0, 2), S("forward", 7, 9)]


def test_effect_inside_a_cut_span_splits_the_hold_at_the_span():
    lines = episode("space space space #:space #:space #:space space space space")
    cut = gui(cut={3, 4, 5})
    assert evidence(lines, cut, OFF, ACTIONS).segments == [S("jump", 0, 8)]
    # a jump counted between cut lines 4 and 5: an effect between kept lines 2 and 6
    lines = bump(lines, CUSTOM + "jump", 4)
    ev = evidence(lines, cut, OFF, ACTIONS)
    assert ev.segments == [S("jump", 0, 2), S("jump", 6, 8)]


def test_r_min_comes_from_the_actions_config():
    lines = episode("w w w # - - w w w")
    cut = gui(cut={3})
    # kept lines w w w - - w w w: a 2-line release, filled iff 2 < r_min
    ev = evidence(lines, cut, OFF, ActionsConfig(turn_deg=10.0, r_min=3))
    assert ev.segments == [S("forward", 0, 8)]
    ev = evidence(lines, cut, OFF, ActionsConfig(turn_deg=10.0, r_min=2))
    assert ev.segments == [S("forward", 0, 2), S("forward", 6, 8)]


def test_segments_merged_by_first_line():
    lines = episode("w w w w w @ @ @ w+M0 w+M0 w+M0 w+M0")
    clicks = [S("right_click_4", 6, 6), S("left_click_1", 6, 6), S("close", 7, 7)]
    ev = evidence(lines, gui(clicks), OFF, ACTIONS)
    assert ev.segments == [
        S("forward", 0, 4),
        S("right_click_4", 6, 6),  # the GUI order within a line is kept
        S("left_click_1", 6, 6),
        S("close", 7, 7),
        S("forward", 8, 11),  # the world order within a line is kept
        S("attack", 8, 11),
    ]


def test_foreign_is_the_union_over_all_lines():
    # q on world line 1; riding between cut lines 4 and 5; a GUI foreign line 8
    lines = bump(episode("w w+q w # # # w @ @ @ w w"), RIDE, 4)
    ev = evidence(lines, gui(cut={3, 4, 5}, foreign={8}), OFF, ACTIONS)
    assert ev.foreign == frozenset({1, 4, 8})


def test_riding_at_the_end_of_a_cut_span_is_on_the_cut_line():
    # the counter grows between cut line 5 and line 6: riding line 5 (not kept line 2)
    lines = bump(episode("w w w # # # w w"), RIDE, 5)
    ev = evidence(lines, gui(cut={3, 4, 5}), OFF, ACTIONS)
    assert ev.foreign == frozenset({5})


@pytest.mark.parametrize(
    "start, expected",
    [(OFF, [F_, F_, T_, T_, T_]), ((T_, F_, 0), [T_, T_, F_, F_, F_])],
)
def test_modes_over_all_lines(start, expected):
    # F3 released at line 1, a cut chest line after a world line: a toggle at frame 2
    lines = episode("w+f3 # # # w")
    ev = evidence(lines, gui(cut={1, 2, 3}), start, ACTIONS)
    assert ev.modes.dtype == np.bool_
    assert ev.modes.tolist() == expected


@pytest.mark.parametrize(
    "start, shown",
    [(OFF, False), ((T_, F_, 0), True), ((F_, T_, 0), True), ((F_, F_, 1), True),
     ((F_, F_, 2), True)],
)  # fmt: skip
def test_modes_start_values(start, shown):
    ev = evidence(episode("w w w"), gui(), start, ACTIONS)
    assert ev.modes.tolist() == [shown] * 3


def test_evidence_carries_lines_cut_and_turn_deg():
    lines = episode("w # w")
    ev = evidence(lines, gui(cut={1}), OFF, ActionsConfig(turn_deg=7.5, r_min=2))
    assert isinstance(ev, Evidence)
    assert list(ev.lines) == lines
    assert ev.cut == frozenset({1})
    assert ev.turn_deg == 7.5


@pytest.mark.parametrize("seed", range(30))
def test_evidence_matches_naive_reference(seed):
    lines, g, start = random_episode(seed, 100)
    ev, tr = evidence(lines, g, start, ACTIONS), truth(lines, g, start, ACTIONS)
    assert kept(ev) == tr.kappa
    assert ev.segments == tr.segs
    assert ev.cut == tr.cut
    assert ev.foreign == tr.foreign
    assert ev.modes.tolist() == tr.modes.tolist()
