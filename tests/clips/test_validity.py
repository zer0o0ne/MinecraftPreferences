"""Validity of an interval (1), of a state (2) and the valid starts (3) of specs/clips.md."""

import numpy as np
import pytest
from vpt.action_lines import bump, episode, repeat

from mcpref.clips.evidence import evidence
from mcpref.clips.validity import interval, valid_starts
from mcpref.vpt.actions import Segment as S

from .clip_oracle import interval_truth, starts_truth, truth
from .clip_world import ACTIONS, OFF, RIDE, gui, random_episode

R, U = "turn_right", "turn_up"
HOLD_ACROSS_CUT = "w w w #:w #:w #:w #:w w w w"  # lines 3-6 cut: kappa = 0 1 2 7 8 9


def test_interval_spans_delta_kept_lines_across_a_cut_span():
    yaw, pitch = [0] * 5 + [20] * 5, [0] * 8 + [-10] * 2
    lines = episode(HOLD_ACROSS_CUT, yaw=yaw, pitch=pitch)
    ev = evidence(lines, gui(cut=range(3, 7)), OFF, ACTIONS)
    assert [interval(ev, [0, 1, 2, 7, 8, 9], p, 2) for p in range(4)] == [
        (["forward"], True),  # lines 0 -> 2
        (["forward", R, R], True),  # 1 -> 7: the yaw step 4 -> 5 lies in the cut span
        (["forward", R, R, U], True),  # 2 -> 8: pitch 0 -> -10
        (["forward", U], True),  # 7 -> 9
    ]


def test_interval_without_actions_is_invalid():
    ev = evidence(episode("- - - - -", yaw=[0, 0, 0, 3, 7]), gui(), OFF, ACTIONS)
    kappa = [0, 1, 2, 3, 4]
    assert interval(ev, kappa, 0, 2) == ([], False)  # n = 0
    assert interval(ev, kappa, 1, 2) == ([], False)  # 3 degrees: no turn
    assert interval(ev, kappa, 2, 2) == ([R], True)  # 7 degrees: n = 1


def test_interval_with_gui_actions_only_is_valid():
    lines = episode("- @ @ @ -")
    ev = evidence(lines, gui([S("left_click_2", 2, 2), S("close", 3, 3)]), OFF, ACTIONS)
    kappa = [0, 1, 2, 3, 4]
    assert interval(ev, kappa, 0, 2) == ([], False)
    assert interval(ev, kappa, 1, 2) == (["left_click_2"], True)
    assert interval(ev, kappa, 2, 2) == (["left_click_2", "close"], True)


@pytest.mark.parametrize(
    "q, valid", [(1, True), (2, False), (3, False), (4, False), (5, True)]
)
def test_foreign_line_counts_on_lines_a_to_b_minus_1(q, valid):
    tokens = ["w"] * 8
    tokens[q] = "w+q"
    ev = evidence(episode(" ".join(tokens)), gui(), OFF, ACTIONS)
    assert interval(ev, list(range(8)), 2, 3) == (["forward"], valid)  # lines 2 -> 5


def test_gui_foreign_line_invalidates_its_intervals():
    ev = evidence(episode("w w @ @ @ w w"), gui([S("left_click_1", 3, 3)], foreign={4}),
                  OFF, ACTIONS)  # fmt: skip
    kappa = list(range(7))
    assert interval(ev, kappa, 2, 2) == (["left_click_1"], True)  # lines 2, 3
    assert interval(ev, kappa, 3, 2) == (["left_click_1"], False)  # lines 3, 4


@pytest.mark.parametrize(
    "ride, valid",
    [(1, [False, True, True]), (2, [True, False, True]), (3, [True, False, True]),
     (6, [True, False, True]), (7, [True, True, False])],
)  # fmt: skip
def test_riding_line_inside_a_cut_span_invalidates(ride, valid):
    # intervals of delta = 1 over the kept lines: 1 -> 2, 2 -> 7 (cut lines 3-6), 7 -> 8
    lines = bump(episode(HOLD_ACROSS_CUT), RIDE, ride)
    ev = evidence(lines, gui(cut=range(3, 7)), OFF, ACTIONS)
    found = [interval(ev, [0, 1, 2, 7, 8, 9], p, 1) for p in (1, 2, 3)]
    assert found == [(["forward"], v) for v in valid]


def test_mode_on_first_middle_or_last_state():
    # F1 pressed at lines 4 and 5: the HUD hidden at frame 6 only
    ev = evidence(episode("w w w w w+f1 w+f1! w w w w w w"), gui(), OFF, ACTIONS)
    # L = 3, delta = 2: states p, p + 2, p + 4; frame 6 is the last state of p = 2, the
    # middle one of p = 4 and the first one of p = 6; p = 3, 5 step over it
    assert valid_starts(ev, list(range(12)), 2, 3) == [0, 1, 3, 5, 7]


def test_states_are_the_kept_lines():
    # lines 2-4 cut; F1 at lines 5 and 6: the HUD hidden at frame 7 = kappa_4 only
    lines = episode("w w #:w #:w #:w w+f1 w+f1! w w w w w w")
    ev = evidence(lines, gui(cut={2, 3, 4}), OFF, ACTIONS)
    kappa = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12]
    assert valid_starts(ev, kappa, 1, 2) == [0, 1, 2, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "tokens, cut, delta, states, expected",
    [(repeat("w", 10), (), 3, 4, [0]), (repeat("w", 9), (), 3, 4, []),
     (repeat("w", 2), (), 1, 2, [0]), ("w", (), 1, 2, []),
     (repeat("w", 12), (), 2, 3, list(range(8))), ("w w w # #", {3, 4}, 1, 3, [0]),
     ("w w w # #", {3, 4}, 1, 4, [])],
)  # fmt: skip
def test_clip_exists_iff_its_last_state_is_kept(tokens, cut, delta, states, expected):
    ev = evidence(episode(tokens), gui(cut=cut), OFF, ACTIONS)
    kappa = [t for t in range(len(ev.lines)) if t not in cut]
    assert valid_starts(ev, kappa, delta, states) == expected


def test_one_invalid_interval_removes_every_clip_containing_it():
    # q at line 4: interval 4 -> 5 invalid; L = 3, delta = 1: clips 3 and 4 contain it
    ev = evidence(episode("w w w w w+q w w w"), gui(), OFF, ACTIONS)
    assert valid_starts(ev, list(range(8)), 1, 3) == [0, 1, 2, 5]


def setup(seed: int):
    lines, g, start = random_episode(seed, 100)
    rng = np.random.default_rng(1000 + seed)
    delta, states = int(rng.integers(1, 4)), int(rng.integers(2, 5))
    return (
        evidence(lines, g, start, ACTIONS),
        truth(lines, g, start, ACTIONS),
        delta,
        states,
    )


@pytest.mark.parametrize("seed", range(30))
def test_valid_starts_match_naive_reference(seed):
    ev, tr, delta, states = setup(seed)
    assert valid_starts(ev, tr.kappa, delta, states) == starts_truth(tr, delta, states)


@pytest.mark.parametrize("seed", range(30))
def test_intervals_match_naive_reference(seed):
    ev, tr, delta, _ = setup(seed)
    ps = range(len(tr.kappa) - delta)
    found = [interval(ev, tr.kappa, p, delta) for p in ps]
    assert found == [interval_truth(tr, p, delta) for p in ps]


@pytest.mark.parametrize("seed", [100, 102, 104, 106])
@pytest.mark.parametrize("delta, states", [(5, 3), (20, 2), (20, 3)])
def test_long_intervals_match_naive_reference(seed, delta, states):
    lines, g, start = random_episode(seed, 600)
    ev, tr = evidence(lines, g, start, ACTIONS), truth(lines, g, start, ACTIONS)
    assert valid_starts(ev, tr.kappa, delta, states) == starts_truth(tr, delta, states)


def test_random_episodes_exercise_every_rule():
    """The naive comparisons above see every cause of (1)-(3)."""
    causes = dict.fromkeys(
        ["n=0", "F cut", "F kept", "valid", "mode", "V", "joined"], 0
    )
    for seed in range(30):
        _, tr, delta, states = setup(seed)
        for p in range(len(tr.kappa) - delta):
            acts, ok = interval_truth(tr, p, delta)
            bad = [
                k for k in range(tr.kappa[p], tr.kappa[p + delta]) if k in tr.foreign
            ]
            causes["n=0"] += not acts
            causes["F cut"] += any(k in tr.cut for k in bad)
            causes["F kept"] += any(k not in tr.cut for k in bad)
            causes["valid"] += ok
        causes["mode"] += int(tr.modes[tr.kappa].sum())
        causes["V"] += bool(starts_truth(tr, delta, states))
        causes["joined"] += sum(
            any(s.first < c < s.last for c in tr.cut) for s in tr.segs
        )
    assert min(causes.values()) > 0, causes
    assert causes["V"] >= 20, causes
