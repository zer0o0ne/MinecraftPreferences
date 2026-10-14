"""`folds` of src/mcpref/recognition/frame_set.py against specs/gui-masks.md (13): a hand case
and the naive reference of masks_oracle.py."""

from collections import Counter

import pytest
from masks_oracle import folds as naive_folds

from mcpref.recognition.frame_set import folds

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
