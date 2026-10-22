"""`weights`, `next_actions`, `targets` of src/mcpref/policy/targets.py against specs/policy.md
(1), (7), (8): a hand-made clip (repeats, a masked action dropped from the target and kept in
the input, $n'_s = 0$, $\\chi^\\pi$ with $n_s \\ne 1$ and $n_{s+1} \\ne 1$, a masked single
action, the last two states), a GUI state, random clips against the transcription
(policy_oracle.py); shapes and dtypes. A and the masks: the approved ones (`masks`)."""

import numpy as np
import pytest

from mcpref.policy.targets import next_actions, targets, weights

from . import policy_oracle as oracle
from .policy_world import HAND, make_clip, random_clips


def index_of(masks) -> dict:
    return {a: i for i, a in enumerate(masks.names)}


def test_weights_by_hand(masks):
    """(1): rows 0..6 the counts of each interval's actions over n (masked or not: hotbar_3
    at state 4, turn_up at state 5, the GUI click of interval 6), none 0; row 7 the one-hot
    of none (index |A|)."""
    i = index_of(masks)
    got = weights(HAND, i)
    assert got.dtype == np.float64 and got.shape == (8, 396)
    want = np.zeros((8, 396))
    rows = [{"forward": 1.0}, {"jump": 1.0}, {"attack": 2 / 3, "turn_left": 1 / 3},
            {"use": 1.0}, {"hotbar_3": 1.0}, {"turn_up": 0.5, "sneak": 0.5},
            {"left_click_3": 1.0}]  # fmt: skip
    for j, row in enumerate(rows):
        for a, v in row.items():
            want[j, i[a]] = v
    want[7, 395] = 1.0
    np.testing.assert_allclose(got, want, rtol=1e-15, atol=0)
    np.testing.assert_allclose(got.sum(1), np.ones(8), rtol=1e-15)


def test_next_actions_by_hand(masks):
    """(7): the action of A_{s+1} for s <= L - 3 = 5 with n_{s+1} = 1 (masked or not), else
    index 0; states 6, 7 (no next interval in the clip) index 0."""
    i = index_of(masks)
    got = next_actions(HAND, i)
    assert got.dtype == np.int64 and got.shape == (8,)
    want = [i["jump"], 0, i["use"], i["hotbar_3"], 0, i["left_click_3"], 0, 0]
    assert got.tolist() == want


def test_targets_by_hand(masks):
    """(8) of states 0..5 from the intervals 1..6 and the masks of states 1..6."""
    i = index_of(masks)
    q, chi = targets(HAND, masks)
    assert q.dtype == np.float64 and q.shape == (6, 395)
    assert chi.dtype == np.bool_ and chi.shape == (6,)
    want = np.zeros((6, 395))
    want[0, i["jump"]] = 1.0  # n_0 = n_1 = n'_0 = 1: chi
    want[1, i["attack"]], want[1, i["turn_left"]] = 2 / 3, 1 / 3  # repeats, n_2 = 3
    want[2, i["use"]] = 1.0  # n_2 = 3 != 1: no chi
    # state 3: hotbar_3 masked at state 4 (hotbar 2): q = 0, n'_3 = 0, no chi
    want[4, i["sneak"]] = 1.0  # turn_up masked at state 5 (pitch -90); n_5 = 2: no chi
    # state 5: left_click_3 masked at the world state 6: q = 0
    np.testing.assert_allclose(q, want, rtol=1e-15, atol=0)
    assert chi.tolist() == [True, False, False, False, False, False]


def test_gui_state_mask(masks):
    """The mask of state s + 1 in a GUI (`inventory`): `close` allowed, `forward` masked; a
    world state after it."""
    clip = make_clip([["jump"], ["close", "forward"], ["close"], ["use"]],
                     pitch=[0.0] * 5, hotbar=[0] * 5, gui=[False, True, True, False, False],
                     classes=[None, "inventory", "inventory", None, None])  # fmt: skip
    i = index_of(masks)
    q, chi = targets(clip, masks)
    want = np.zeros((3, 395))
    want[0, i["close"]] = 1.0  # forward masked at the GUI state 1; n_1 = 2: no chi
    want[1, i["close"]] = 1.0  # close allowed at the GUI state 2; n_1 = 2: no chi
    want[2, i["use"]] = (
        1.0  # use allowed at the world state 3; n_2 = n_3 = n'_2 = 1: chi
    )
    np.testing.assert_allclose(q, want, rtol=1e-15, atol=0)
    assert chi.tolist() == [False, False, True]


@pytest.mark.parametrize("seed", range(4))
def test_random_clips_against_transcription(masks, seed):
    """World states with pitches at and inside the limits, hotbars 0..8, masked and GUI
    actions in the intervals: (1), (7), (8) as transcribed."""
    names, i = masks.names, index_of(masks)
    for clip in random_clips(seed, 5, 6):
        np.testing.assert_allclose(weights(clip, i), oracle.input_weights(clip, names), rtol=1e-15, atol=0)  # fmt: skip
        assert next_actions(clip, i).tolist() == oracle.next_actions(clip, names)
        q, chi = targets(clip, masks)
        want_q, want_chi = oracle.targets(clip, names)
        np.testing.assert_allclose(q, want_q, rtol=1e-15, atol=0)
        assert chi.tolist() == want_chi.tolist()


def test_shortest_clip(masks):
    """L = 3: one target (state 0); next_actions rows 1, 2 index 0."""
    clip = make_clip([["attack"], ["use"]], [0.0] * 3, [0] * 3)
    i = index_of(masks)
    q, chi = targets(clip, masks)
    assert q.shape == (1, 395) and q[0, i["use"]] == 1.0 and chi.tolist() == [True]
    assert next_actions(clip, i).tolist() == [i["use"], 0, 0]


def test_index_maps_names(masks):
    """`weights` and `next_actions` place the actions by the given index (here A reversed)."""
    names = tuple(reversed(masks.names))
    i = {a: k for k, a in enumerate(names)}
    np.testing.assert_allclose(weights(HAND, i), oracle.input_weights(HAND, names))
    assert next_actions(HAND, i).tolist() == oracle.next_actions(HAND, names)
