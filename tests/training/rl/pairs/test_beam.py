"""Equations (5)-(6): polarity and beam discovery on hand-built History graphs."""

import numpy as np
import torch

from mcpref.training.rl.pairs.cycle import sample_pairs
from mcpref.training.rl.pairs.types import StateRef
from training.rl.pairs.helpers import (
    Encoder,
    Preference,
    comparison,
    config,
    from_dict,
    vertex,
    work,
    write_history,
)


def edge(records, a, b, p, inc):
    records[a]["comparisons"].append(comparison(a, b, p, inc, 1))
    records[b]["comparisons"].append(comparison(b, a, 1 - p, inc, 1))


def graph(refs):
    return {r: vertex(r, [float(100 + i)]) for i, r in enumerate(refs)}


def discover(tmp_path, records, **changes):
    cfg = config(tmp_path, sample_states=1, temporal_per_state=0,
                 nearest_per_state=1, **changes)  # fmt: skip
    write_history(
        cfg, 0, sorted(records.values(), key=lambda v: (v["episode"], v["index"]))
    )
    seed = StateRef("new", 0)
    out = sample_pairs(
        cfg,
        1,
        [seed],
        Encoder({seed: (0.0,)}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    return [from_dict(c["b"]) for c in work(out)[0]["comparisons"]]


def test_polarity_extremes_neutral_and_stable_ties(tmp_path):
    a, neutral, negative, incomparable, positive = [
        StateRef("old", i) for i in range(5)
    ]
    records = graph([a, neutral, negative, incomparable, positive])
    records[a]["key"] = [0.0]
    edge(records, a, neutral, 0.5, 0.0)
    edge(records, a, negative, 0.0, 0.0)
    edge(records, a, incomparable, 0.5, 1.0)
    edge(records, a, positive, 1.0, 0.0)
    assert discover(tmp_path, records, branch=4, beam_width=4, beam_depth=1) == [
        a,
        negative,
        incomparable,
        positive,
        neutral,
    ]


def test_beam_branch_limit_then_global_width(tmp_path):
    a, b, c, d = [StateRef("old", i) for i in range(4)]
    records = graph([a, b, c, d])
    records[a]["key"] = [0.0]
    for partner, p in [(b, 0.95), (c, 0.9), (d, 0.85)]:
        edge(records, a, partner, p, 0.0)
    assert discover(tmp_path, records, branch=2, beam_width=1, beam_depth=1) == [a, b]


def test_beam_uses_best_discovery_polarity_across_frontier(tmp_path):
    a, b, c, d = [StateRef("old", i) for i in range(4)]
    records = graph([a, b, c, d])
    records[a]["key"] = [0.0]
    edge(records, a, b, 0.9, 0.0)
    edge(records, a, c, 0.8, 0.0)
    edge(records, b, d, 0.8, 0.0)  # J = 0.6
    edge(records, c, d, 0.975, 0.0)  # J = 0.95, discovered second but wins
    assert discover(tmp_path, records, branch=2, beam_width=2, beam_depth=2) == [
        a,
        b,
        c,
        d,
    ]


def test_beam_depth_excludes_later_frontiers(tmp_path):
    a, b, c = [StateRef("old", i) for i in range(3)]
    records = graph([a, b, c])
    records[a]["key"] = [0.0]
    edge(records, a, b, 0.9, 0.0)
    edge(records, b, c, 0.9, 0.0)
    assert discover(tmp_path, records, branch=2, beam_width=2, beam_depth=1) == [a, b]


def test_beam_early_stop_and_visited_deduplication(tmp_path):
    a, b, c = [StateRef("old", i) for i in range(3)]
    records = graph([a, b, c])
    records[a]["key"] = [0.0]
    edge(records, a, b, 0.9, 0.0)
    edge(records, b, c, 0.9, 0.0)
    edge(records, c, a, 0.9, 0.0)
    assert discover(tmp_path, records, branch=3, beam_width=3, beam_depth=8) == [
        a,
        b,
        c,
    ]


def test_beam_excludes_current_episode_and_discarded_candidates(tmp_path):
    a, b, discarded = [StateRef("old", i) for i in range(3)]
    current = StateRef("new", 99)
    records = graph([a, b, discarded, current])
    records[a]["key"] = [0.0]
    edge(records, a, current, 1.0, 0.0)
    edge(records, a, discarded, 0.9, 0.0)
    edge(records, a, b, 0.8, 0.0)
    edge(records, b, current, 1.0, 0.0)
    assert discover(tmp_path, records, branch=2, beam_width=1, beam_depth=3) == [
        a,
        discarded,
    ]
