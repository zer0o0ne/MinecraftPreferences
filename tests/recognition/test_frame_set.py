"""`folds` and `split_draw` of src/mcpref/recognition/frame_set.py against specs/gui-masks.md
(13) and the split: a hand case, the naive reference of masks_oracle.py, numpy's own draws;
the helpers of src/mcpref/gui/label_files.py this revision removes."""

import json
from collections import Counter

import numpy as np
import pytest
from masks_oracle import folds as naive_folds

from mcpref.gui import label_files
from mcpref.recognition.frame_set import folds, split_draw

EPISODES = [f"data/7.0/ep{7 * e % 23:02d}.mp4" for e in range(23)]  # not sorted


def test_folds_hand_case():
    """PCG64(7).permutation(5) = [2, 0, 4, 1, 3]: e2 -> 0, e0 -> 1, e4 -> 0, e1 -> 1, e3 -> 0."""
    relpaths = ["e0", "e1", "e2", "e3", "e4"]
    assert folds(relpaths, 2, 7) == {"e0": 1, "e1": 1, "e2": 0, "e3": 0, "e4": 0}


CASES = [(n, k) for n in (1, 2, 3, 5, 6, 9, 23) for k in (2, 3, 5)]


@pytest.mark.parametrize("n, k", CASES)
@pytest.mark.parametrize("seed", [0, 1, 230])
def test_folds_naive_reference(n, k, seed):
    """(13): episode pi[r] gets fold r mod K; n < K leaves folds empty, n = K, n a multiple of K."""
    relpaths = EPISODES[:n]
    got = folds(relpaths, k, seed)
    assert got == naive_folds(relpaths, k, seed)
    assert all(type(v) is int and 0 <= v < k for v in got.values())


@pytest.mark.parametrize("n, k", [(23, 2), (23, 5), (9, 3), (3, 5)])
def test_folds_balanced(n, k):
    """r mod K over r = 0..n-1: fold sizes ceil(n / K) or floor(n / K), every episode once."""
    got = folds(EPISODES[:n], k, 1)
    assert sorted(got) == sorted(EPISODES[:n])
    sizes = Counter(got.values())
    assert [sizes[l] for l in range(k)] == [n // k + (l < n % k) for l in range(k)]


def test_folds_fixed_seed():
    """The same seed gives the same folds; another seed other folds."""
    assert folds(EPISODES, 5, 1) == folds(list(EPISODES), 5, 1)
    assert folds(EPISODES, 5, 1) != folds(EPISODES, 5, 2)


PROBS = [[0.6, 0.2, 0.2], [0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]]


@pytest.mark.parametrize("probs", PROBS)
@pytest.mark.parametrize("seed", [0, 230])
def test_split_draw_is_one_choice(probs, seed):
    """Split: each call is one draw rng.choice(3, p=probs) of the generator it is given."""
    rng, ref = (np.random.Generator(np.random.PCG64(seed)) for _ in range(2))
    got = [split_draw(rng, probs) for _ in range(40)]
    assert got == [int(ref.choice(3, p=probs)) for _ in range(40)]
    assert all(type(d) is int and probs[d] > 0 for d in got)


@pytest.mark.parametrize("k", [0, 1, 7, 20])
def test_split_draw_continues_from_a_restored_state(k):
    """(I): the generator's bit_generator.state after k draws, stored as JSON and set on a
    Generator(PCG64()), continues the draws exactly."""
    probs = [0.6, 0.2, 0.2]
    rng = np.random.Generator(np.random.PCG64(0))
    whole = [split_draw(rng, probs) for _ in range(30)]
    rng = np.random.Generator(np.random.PCG64(0))
    first = [split_draw(rng, probs) for _ in range(k)]
    restored = np.random.Generator(np.random.PCG64())
    restored.bit_generator.state = json.loads(json.dumps(rng.bit_generator.state))
    assert first + [split_draw(restored, probs) for _ in range(30 - k)] == whole


def test_label_files_helpers_removed():
    """Interface: `ok_frames` and `read_jsonl` are gone from src/mcpref/gui/label_files.py."""
    assert not hasattr(label_files, "ok_frames")
    assert not hasattr(label_files, "read_jsonl")
