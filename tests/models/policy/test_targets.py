"""Policy inputs use prior actions; targets use source-state masks and own intervals.

Hand examples cover repeated, masked and GUI actions, terminal states and validity;
random clips check shapes and values against the independent transcription.
"""

import numpy as np
import pytest

from mcpref.models.policy.targets import next_actions, targets, weights
from models.policy import policy_oracle as oracle
from models.policy.policy_world import HAND, make_clip, random_clips


def index_of(masks) -> dict:
    return {a: i for i, a in enumerate(masks.names)}


def test_weights_by_hand(masks):
    """(1): row 0 has no action; rows 1..7 contain prior-interval frequencies."""
    i = index_of(masks)
    got = weights(HAND, i)
    assert got.dtype == np.float64 and got.shape == (8, 396)
    want = np.zeros((8, 396))
    rows = [{"forward": 1.0}, {"jump": 1.0}, {"attack": 2 / 3, "turn_left": 1 / 3},
            {"use": 1.0}, {"hotbar_3": 1.0}, {"turn_up": 0.5, "sneak": 0.5},
            {"left_click_3": 1.0}]  # fmt: skip
    for j, row in enumerate(rows, start=1):
        for a, v in row.items():
            want[j, i[a]] = v
    want[0, 395] = 1.0
    np.testing.assert_allclose(got, want, rtol=1e-15, atol=0)
    np.testing.assert_allclose(got.sum(1), np.ones(8), rtol=1e-15)


def test_next_actions_by_hand(masks):
    """(7): each interval's single action; the terminal row has placeholder 0."""
    i = index_of(masks)
    got = next_actions(HAND, i)
    assert got.dtype == np.int64 and got.shape == (8,)
    want = [
        i["forward"],
        i["jump"],
        0,
        i["use"],
        i["hotbar_3"],
        0,
        i["left_click_3"],
        0,
    ]
    assert got.tolist() == want


def test_targets_by_hand(masks):
    """(8) of states 0..6 from intervals 0..6 and their source-state masks."""
    i = index_of(masks)
    q, chi = targets(HAND, masks)
    assert q.dtype == np.float64 and q.shape == (7, 395)
    assert chi.dtype == np.bool_ and chi.shape == (7,)
    want = np.zeros((7, 395))
    want[0, i["forward"]] = 1.0
    want[1, i["jump"]] = 1.0
    want[2, i["attack"]], want[2, i["turn_left"]] = 2 / 3, 1 / 3
    want[3, i["use"]] = 1.0  # Prior interval has three actions; this shift is valid.
    # hotbar_3 is masked at state 4; left_click_3 is masked at world state 6.
    want[5, i["sneak"]] = 1.0  # Filtering leaves one action, but n_5 = 2: no shift.
    np.testing.assert_allclose(q, want, rtol=1e-15, atol=0)
    assert chi.tolist() == [True, True, False, True, False, False, False]


def test_gui_state_mask(masks):
    """The mask of state s in a GUI (`inventory`): `close` allowed, `forward` masked; a
    world state after it."""
    clip = make_clip([["jump"], ["close", "forward"], ["close"], ["use"]],
                     pitch=[0.0] * 5, hotbar=[0] * 5, gui=[False, True, True, False, False],
                     classes=[None, "inventory", "inventory", None, None])  # fmt: skip
    i = index_of(masks)
    q, chi = targets(clip, masks)
    want = np.zeros((4, 395))
    want[0, i["jump"]] = 1.0
    want[1, i["close"]] = 1.0  # forward masked; two actual actions disallow shift.
    want[2, i["close"]] = 1.0
    want[3, i["use"]] = 1.0
    np.testing.assert_allclose(q, want, rtol=1e-15, atol=0)
    assert chi.tolist() == [True, False, True, True]


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
    """L = 3: both transitions are targets; only terminal state 2 is a placeholder."""
    clip = make_clip([["attack"], ["use"]], [0.0] * 3, [0] * 3)
    i = index_of(masks)
    q, chi = targets(clip, masks)
    assert q.shape == (2, 395) and q[0, i["attack"]] == q[1, i["use"]] == 1.0
    assert chi.tolist() == [True, True]
    assert next_actions(clip, i).tolist() == [i["attack"], i["use"], 0]


def test_index_maps_names(masks):
    """`weights` and `next_actions` place the actions by the given index (here A reversed)."""
    names = tuple(reversed(masks.names))
    i = {a: k for k, a in enumerate(names)}
    np.testing.assert_allclose(weights(HAND, i), oracle.input_weights(HAND, names))
    assert next_actions(HAND, i).tolist() == oracle.next_actions(HAND, names)
