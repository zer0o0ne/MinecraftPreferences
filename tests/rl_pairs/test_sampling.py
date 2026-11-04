"""Equations (1)-(4): auxiliary scores, seed/temporal draws, and nearest alternatives."""

import json

import numpy as np
import pytest
import torch

from mcpref.rl_pairs.cycle import sample_pairs
from mcpref.rl_pairs.sample import choose_seeds, seed_scores, temporal
from mcpref.rl_pairs.types import StateRef

from .helpers import (
    Encoder,
    Preference,
    config,
    from_dict,
    shards,
    vertex,
    vertices,
    work,
    write_history,
)


def score_oracle(p, inc):
    return max(inc, (1 - inc) * abs(2 * p - 1))


def test_seed_scores_immediate_predecessor_hand_cases(tmp_path):
    states = [StateRef("episode", i) for i in range(5)]
    encode = Encoder({r: (float(i),) for i, r in enumerate(states)})
    cases = [(0.9, 0.1), (0.1, 0.1), (0.3, 0.8), (0.5, 0.2)]
    results = {((float(i - 1),), (float(i),)): cases[i - 1] for i in range(1, 5)}
    preference = Preference(results).train()
    out = seed_scores(states, encode, preference, tmp_path / "scores")
    scores = [score for shard in shards(out, "scores.json") for score in shard]
    expected = [0.0] + [score_oracle(*case) for case in cases]
    assert scores == pytest.approx(expected)
    assert preference.calls == [((float(i - 1),), (float(i),)) for i in range(1, 5)]
    assert preference.training and preference.dummy.grad is None
    assert {p.name for p in tmp_path.iterdir()} == {"scores"}


def write_scores(directory, values):
    shard = directory / "000000"
    shard.mkdir(parents=True)
    (shard / "scores.json").write_text(json.dumps(values))


def draw_oracle(scores, count, seed):
    rng = np.random.default_rng(seed)
    positive = [i for i, score in enumerate(scores) if score > 0]
    zero = [i for i, score in enumerate(scores) if score == 0]
    n = min(count, len(scores))
    n_positive = min(n, len(positive))
    selected = []
    if n_positive:
        probabilities = np.array([scores[i] for i in positive], dtype=np.float64)
        probabilities /= probabilities.sum()
        selected.extend(rng.choice(np.array(positive), size=n_positive, replace=False,
                                   p=probabilities, axis=0, shuffle=True).tolist())  # fmt: skip
    if n - n_positive:
        selected.extend(rng.choice(np.array(zero), size=n - n_positive, replace=False,
                                   p=None, axis=0, shuffle=True).tolist())  # fmt: skip
    return selected


@pytest.mark.parametrize(
    "scores,count,seed",
    [
        ([0.0, 0.2, 0.0, 0.8, 0.0], 4, 8),
        ([0.1, 0.2, 0.3], 3, 9),
        ([0.0, 0.0, 0.0, 0.0], 3, 10),
        ([0.5, 0.0, 0.5, 0.0], 1, 11),
        ([0.0, 0.4, 0.0], 5, 12),
    ],
    ids=["zero-fill", "no-zero-fill", "all-zero", "weighted", "count-clamped"],
)
def test_choose_seeds_matches_numpy_draw_branch(tmp_path, scores, count, seed):
    write_scores(tmp_path / "scores", scores)
    actual = choose_seeds(tmp_path / "scores", count, np.random.default_rng(seed))
    assert actual == draw_oracle(scores, count, seed)
    assert len(actual) == len(set(actual)) == min(count, len(scores))


class NoChoice:
    def choice(self, *args, **kwargs):
        raise AssertionError("zero-sized draw called the RNG")


def test_zero_sized_seed_draw_skips_rng(tmp_path):
    write_scores(tmp_path / "scores", [0.0, 1.0])
    assert choose_seeds(tmp_path / "scores", 0, NoChoice()) == []


def test_zero_score_fill_is_skipped_when_positive_draw_reaches_count(tmp_path):
    class CountingChoice:
        def __init__(self):
            self.rng, self.calls = np.random.default_rng(3), 0

        def choice(self, *args, **kwargs):
            self.calls += 1
            return self.rng.choice(*args, **kwargs)

    write_scores(tmp_path / "scores", [0.5, 0.5, 0.0])
    rng = CountingChoice()
    assert choose_seeds(tmp_path / "scores", 2, rng) == [0, 1]
    assert rng.calls == 1


def temporal_oracle(seeds, seed, count, rng):
    others = [s for s in seeds if s != seed]
    if not others or not count:
        return []
    weights = np.array([abs(seed - s) for s in others], dtype=np.float64)
    weights /= weights.sum()
    return rng.choice(np.array(others), size=min(count, len(others)), replace=False,
                      p=weights, axis=0, shuffle=True).tolist()  # fmt: skip


@pytest.mark.parametrize(
    "seeds,seed,count",
    [
        ([0, 2, 5, 9], 2, 2),
        ([4, 1, 8], 4, 8),
        ([0, 3], 3, 1),
    ],
)
def test_temporal_probabilities_count_and_returned_order(seeds, seed, count):
    rng = np.random.default_rng(91)
    expected = temporal_oracle(seeds, seed, count, np.random.default_rng(91))
    assert temporal(seeds, seed, count, rng) == expected
    assert set(expected) <= set(seeds) - {seed}


@pytest.mark.parametrize("seeds,seed,count", [([2], 2, 1), ([], 1, 1), ([1, 3], 1, 0)])
def test_temporal_empty_draw_skips_rng(seeds, seed, count):
    assert temporal(seeds, seed, count, NoChoice()) == []


def test_nearest_uses_stale_keys_distance_and_stable_ties(tmp_path):
    cfg = config(tmp_path, sample_states=1, temporal_per_state=0,
                 nearest_per_state=3, beam_depth=0)  # fmt: skip
    states = [StateRef("new", i) for i in range(3)]
    old = [StateRef("old", i) for i in range(4)]
    records = [
        vertex(old[0], [0, 2]),
        vertex(old[1], [9, 0]),
        vertex(old[2], [1.5, 0]),
        vertex(old[3], [0, 2]),
        vertex(StateRef("new", 99), [0, 0]),
    ]
    write_history(cfg, 0, records)
    encode = Encoder({r: (0.0, 0.0) for r in states})
    preference = Preference().train()
    out = sample_pairs(
        cfg,
        1,
        states,
        encode,
        preference,
        torch.tensor([2.0, 1.0]),
        np.random.default_rng(4),
    )
    assert [from_dict(c["b"]) for c in work(out)[0]["comparisons"]] == [
        old[0],
        old[3],
        old[2],
    ]
    stored = {from_dict(v): v["key"] for v in vertices(tmp_path / "history", 1)}
    assert all(
        stored[r] == next(v["key"] for v in records if from_dict(v) == r)
        for r in [*old, StateRef("new", 99)]
    )
    fresh = {r for r in stored if r.episode == "new" and r.index != 99}
    assert len(fresh) == 1 and fresh <= set(states)
