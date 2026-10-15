"""Sampling of non-overlapping clips (4) and the content of a clip (5) of specs/clips.md."""

import numpy as np
import pytest
from vpt.action_lines import episode, repeat

from mcpref.clips.evidence import evidence
from mcpref.clips.sample import Clip, sample
from mcpref.vpt.actions import Segment as S

from .clip_oracle import clip_truth, draws_truth, starts_truth, truth
from .clip_world import ACTIONS, OFF, Log, gui, random_episode

R, U, D = "turn_right", "turn_up", "turn_down"
FIRST = {0: 2, 1: 1, 11: 0}  # seed -> np.random.default_rng(seed).integers(3)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def hold(T: int):
    """W held on T lines: one forward segment, every interval valid; (evidence, truth)."""
    lines = episode(repeat("w", T))
    return evidence(lines, gui(), OFF, ACTIONS), truth(lines, gui(), OFF, ACTIONS)


def test_first_draws_of_the_seeds_used_here():
    assert {s: int(rng(s).integers(3)) for s in FIRST} == FIRST
    h = rng(1)
    assert [int(h.integers(n)) for n in (3, 2, 1)] == [1, 1, 0]


def test_clips_in_the_order_taken():
    ev, tr = hold(20)
    starts = [0, 3, 6, 9, 12, 15, 18]  # L = 2, delta = 1: disjoint spans [p, p + 1]
    g, log = rng(3), Log()
    clips = sample(ev, tr.kappa, starts, 1, 2, 7, g, log)
    h, w = rng(3), list(starts)
    order = [w.pop(h.integers(len(w))) for _ in starts]  # a draw removes its start only
    assert order != starts
    assert [c.p for c in clips] == order
    assert log.calls == [[p, p + 1] for p in order]
    assert g.bit_generator.state == h.bit_generator.state


@pytest.mark.parametrize("seed, taken", [(1, [4]), (11, [0, 8]), (0, [8, 0])])
def test_spans_touching_at_one_line_overlap(seed, taken):
    # L = 3, delta = 2: the spans 0-4, 4-8, 8-12 touch at lines 4 and 8
    ev, tr = hold(13)
    clips = sample(ev, tr.kappa, [0, 4, 8], 2, 3, 3, rng(seed), Log())
    assert [c.p for c in clips] == taken


def test_disjoint_spans_are_all_taken():
    # L = 3, delta = 2: the spans 0-4, 5-9, 10-14 share no line
    ev, tr = hold(15)
    clips = sample(ev, tr.kappa, [0, 5, 10], 2, 3, 3, rng(1), Log())
    assert [c.p for c in clips] == [5, 10, 0]  # draws 1 of 3, 1 of 2, 0 of 1


def test_span_includes_the_cut_lines_between_its_states():
    # lines 3, 4 cut; L = 2, delta = 1: spans 1-2, 2-5, 5-6; the first draw takes 2
    lines = episode("w w w #:w #:w w w w")
    ev = evidence(lines, gui(cut={3, 4}), OFF, ACTIONS)
    kappa = [0, 1, 2, 5, 6, 7]
    assert [c.p for c in sample(ev, kappa, [1, 2, 3], 1, 2, 3, rng(1), Log())] == [2]
    assert [c.p for c in sample(ev, kappa, [1, 2, 3], 1, 2, 3, rng(11), Log())] == [
        1,
        3,
    ]


def test_rejected_clip_removes_only_its_start():
    # L = 3, delta = 1: spans 0-2, 2-4, 4-6; the clip at 2, drawn first, overlaps both others
    ev, tr = hold(7)
    log = Log(reject={2})
    clips = sample(ev, tr.kappa, [0, 2, 4], 1, 3, 3, rng(1), log)
    assert log.calls[0] == [2, 3, 4]
    assert sorted(c.p for c in clips) == [0, 4]
    assert len(log.calls) == 3


def test_stops_after_m_clips():
    ev, tr = hold(15)
    g, log = rng(5), Log()
    clips = sample(ev, tr.kappa, [0, 3, 6, 9, 12], 1, 2, 2, g, log)
    h = rng(5)
    taken = draws_truth(tr, [0, 3, 6, 9, 12], 1, 2, 2, h, Log())
    assert [c.p for c in clips] == taken and len(taken) == 2
    assert len(log.calls) == 2
    assert g.bit_generator.state == h.bit_generator.state  # two draws


def test_stops_when_no_start_remains():
    # L = 3, delta = 1: the spans of 0, 1, 2 all overlap: one clip after one draw
    ev, tr = hold(5)
    g, h = rng(7), rng(7)
    clips = sample(ev, tr.kappa, [0, 1, 2], 1, 3, 5, g, Log())
    h.integers(3)
    assert len(clips) == 1
    assert g.bit_generator.state == h.bit_generator.state


def test_every_clip_rejected():
    ev, tr = hold(10)
    g, h, log = rng(8), rng(8), Log(reject={0, 3, 6})
    assert sample(ev, tr.kappa, [0, 3, 6], 1, 2, 3, g, log) == []
    assert sorted(log.calls) == [[0, 1], [3, 4], [6, 7]]  # each start drawn once
    for n in (3, 2, 1):
        h.integers(n)
    assert g.bit_generator.state == h.bit_generator.state


def test_no_starts_no_draw():
    ev, tr = hold(4)
    g, log = rng(9), Log()
    assert sample(ev, tr.kappa, [], 1, 2, 3, g, log) == []
    assert log.calls == []
    assert g.bit_generator.state == rng(9).bit_generator.state


def test_starts_and_kappa_are_read_only_sequences():
    ev, tr = hold(13)
    starts = [0, 4, 8]
    clips = sample(ev, tr.kappa, starts, 2, 3, 3, rng(11), Log())
    assert starts == [0, 4, 8]
    assert sample(ev, tuple(tr.kappa), tuple(starts), 2, 3, 3, rng(11), Log()) == clips


# lines 3, 4 cut; a chest at 7-9 (line 7 its first line); the player inventory at 12-13
GUI_EPISODE = "w w w #:w #:w w w # # # w w @ @"
YAW = [0] * 4 + [30] * 10  # +30 between cut lines 3 and 4
PITCH = [0, 0, 5, 5, 5, 5, 17, 17, 17, 17, 17, 17, -25, -25]
CLICKS = [S("left_click_5", 8, 8), S("close", 9, 9), S("left_click_1", 12, 12)]
CLIPS = {
    # states 1, 6, 9, 12; intervals I = {1, 2, 5}, {6, 7, 8}, {9, 10, 11}
    1: Clip(p=1, states=[1, 6, 9, 12],
            actions=[["forward", R, R, R, D, D], ["forward", "left_click_5"],
                     ["close", "forward", U, U, U, U]],
            n=[6, 2, 6], gui=[False, False, True, True], pitch=[0.0, 17.0, 17.0, -25.0]),
    # states 2, 7, 10, 13 (the last kept line); I = {2, 5, 6}, {7, 8, 9}, {10, 11, 12}
    2: Clip(p=2, states=[2, 7, 10, 13],
            actions=[["forward", R, R, R, D], ["left_click_5", "close"],
                     ["forward", "left_click_1", U, U, U, U]],
            n=[5, 2, 6], gui=[False, False, False, True], pitch=[5.0, 17.0, 17.0, -25.0]),
}  # fmt: skip


@pytest.mark.parametrize("p", [1, 2])
def test_clip_carries_actions_n_gui_and_pitch(p):
    lines = episode(GUI_EPISODE, yaw=YAW, pitch=PITCH)
    ev = evidence(lines, gui(CLICKS, cut={3, 4}), OFF, ACTIONS)
    kappa = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    log = Log()
    assert sample(ev, kappa, [p], 3, 4, 1, rng(0), log) == [CLIPS[p]]
    assert log.calls == [CLIPS[p].states]


def test_gui_flag_follows_the_first_line_rule():
    # a chest from line 0, world, a chest from line 3, world, the player inventory from 6
    lines = episode("# # w # # w @ @", pitch=[10.0 * t for t in range(8)])
    ev = evidence(lines, gui(), OFF, ACTIONS)
    clips = sample(ev, list(range(8)), [0, 2, 4, 6], 1, 2, 4, rng(0), Log())
    assert {c.p: c.gui for c in clips} == {
        0: [True, True],  # line 0 opens no span after a world line
        2: [False, False],  # line 3: the first line of a chest
        4: [True, False],
        6: [True, True],  # line 6: the first line of the player inventory
    }


def check_against_naive(episode_of, delta, states, m, reject, seed):
    """sample vs the naive (4), (5) on the valid starts of `episode_of`: clips, accept
    calls and the generator's state after the call."""
    lines, g, start = episode_of
    ev, tr = evidence(lines, g, start, ACTIONS), truth(lines, g, start, ACTIONS)
    starts = starts_truth(tr, delta, states)
    got, ref, gen, h = Log(reject), Log(reject), rng(seed), rng(seed)
    clips = sample(ev, tr.kappa, starts, delta, states, m, gen, got)
    taken = draws_truth(tr, starts, delta, states, m, h, ref)
    assert clips == [Clip(**clip_truth(tr, p, delta, states)) for p in taken]
    assert got.calls == ref.calls
    assert gen.bit_generator.state == h.bit_generator.state


@pytest.mark.parametrize("seed", range(30))
def test_sample_matches_naive_reference(seed):
    r = rng(2000 + seed)
    delta, states = int(r.integers(1, 4)), int(r.integers(2, 5))
    m, reject = int(r.choice([1, 2, 5, 50])), r.choice(100, 30, replace=False).tolist()
    check_against_naive(random_episode(seed, 100), delta, states, m, reject, seed)


@pytest.mark.parametrize("seed", [100, 102, 104, 106])
def test_long_episode_sample_matches_naive_reference(seed):
    # delta = 20 on 600 lines; m = 50 exceeds what fits: sampling ends on an empty W
    reject = range(0, 600, 7)
    check_against_naive(random_episode(seed, 600), 20, 3, 50, reject, seed)
