"""Equations (8)-(9): hand thresholds and current-cycle graph correction."""

import json
import math
from fractions import Fraction

import pytest

from mcpref.graphs.feedback import removed_arcs
from mcpref.rl_pairs.label import correct, threshold
from mcpref.rl_pairs.types import StateRef

from .helpers import config, items, pair, raw, write_work


def threshold_oracle(a, b, cfg):
    if a.episode != b.episode:
        return cfg.threshold_scale * cfg.base_threshold
    distance = abs(a.index - b.index)
    logit = math.log(cfg.base_threshold / (1 - cfg.base_threshold))
    sigmoid = 1 / (1 + math.exp(-(logit - cfg.temporal_penalty / distance)))
    return cfg.threshold_scale * sigmoid


@pytest.mark.parametrize("distance", [1, 2, 100])
def test_temporal_threshold_close_distant_and_by_hand(tmp_path, distance):
    cfg = config(tmp_path, base_threshold=0.6, temporal_penalty=1.7,
                 threshold_scale=0.8)  # fmt: skip
    a, b = StateRef("e", 10), StateRef("e", 10 + distance)
    assert threshold(a, b, cfg) == threshold_oracle(a, b, cfg)


def test_cross_episode_and_zero_penalty_thresholds(tmp_path):
    cfg = config(tmp_path, base_threshold=0.6, temporal_penalty=0.0,
                 threshold_scale=0.8)  # fmt: skip
    assert threshold(StateRef("a", 0), StateRef("b", 100), cfg) == pytest.approx(0.48)
    assert threshold(StateRef("e", 0), StateRef("e", 1), cfg) == pytest.approx(0.48)


def test_threshold_is_strictly_below_scale_cap(tmp_path):
    cfg = config(tmp_path, base_threshold=0.999999, temporal_penalty=0.0,
                 threshold_scale=0.7)  # fmt: skip
    assert 0 < threshold(StateRef("e", 0), StateRef("e", 1), cfg) < 0.7
    assert 0 < threshold(StateRef("a", 0), StateRef("b", 1), cfg) < 0.7


def run_correction(tmp_path, comparisons, **changes):
    cfg = config(tmp_path, **changes)
    work = tmp_path / "work"
    write_work(work, [comparisons], StateRef("seed", 0))
    out = tmp_path / "corrected"
    assert correct(work, out, cfg) == len(comparisons)
    graph = json.loads((out / "graph.json").read_text())
    return out, graph


def test_strict_incomparability_verdict_preserves_soft_probability_and_cost(tmp_path):
    a, b = StateRef("e", 0), StateRef("e", 1)
    cfg = config(tmp_path)
    at_threshold = threshold(a, b, cfg)
    out, graph = run_correction(tmp_path, [raw(a, b, 0.75, at_threshold)])
    assert items(out) == [{"a": {"episode": "e", "index": 0},
                           "b": {"episode": "e", "index": 1},
                           "labels": {"p": 0.75, "inc": 0}}]  # fmt: skip
    assert graph["comparisons"][0]["w"] == 0.75
    assert graph["costs"] == [str(Fraction(str(0.75)) - Fraction(1, 2))]


def test_cycle_correction_labels_and_graph_order(tmp_path):
    a, b, c, d, e = [StateRef("e", i) for i in range(5)]
    comparisons = [raw(a, b, 0.75, 0.0), raw(b, c, 0.75, 0.0),
                   raw(c, a, 0.75, 0.0), raw(d, e, 0.2, 1.0)]  # fmt: skip
    out, graph = run_correction(tmp_path, comparisons)
    assert graph["vertices"] == [{"episode": "e", "index": i} for i in range(3)]
    assert graph["arcs"] == [[0, 1], [1, 2], [2, 0]]
    assert graph["costs"] == ["1/4", "1/4", "1/4"]
    assert graph["removed"] == removed_arcs(3, [(0, 1), (1, 2), (2, 0)],
                                             [Fraction(1, 4)] * 3) == [2]  # fmt: skip
    result = items(out)
    assert [pair(item) for item in result] == [
        (("e", 0), ("e", 1)),
        (("e", 1), ("e", 2)),
        (("e", 2), ("e", 0)),
        (("e", 3), ("e", 4)),
    ]
    assert [item["labels"] for item in result] == [
        {"p": 0.75, "inc": 0},
        {"p": 0.75, "inc": 0},
        {"p": None, "inc": 1},
        {"p": None, "inc": 1},
    ]


@pytest.mark.parametrize("p,expected", [(0.25, (1, 0)), (0.5, (0, 1))])
def test_preferred_first_orientation_and_half_tie(tmp_path, p, expected):
    a, b = StateRef("e", 0), StateRef("e", 1)
    out, graph = run_correction(tmp_path, [raw(a, b, p, 0.0)])
    assert pair(items(out)[0]) == (("e", expected[0]), ("e", expected[1]))
    assert items(out)[0]["labels"] == {"p": max(p, 1 - p), "inc": 0}
    assert graph["comparisons"][0]["w"] == max(p, 1 - p)


def test_removed_arc_keeps_preferred_first_orientation(tmp_path):
    a, b = StateRef("e", 0), StateRef("e", 1)
    out, graph = run_correction(tmp_path, [raw(a, b, 0.25, 0.0),
                                           raw(b, a, 0.25, 0.0)])  # fmt: skip
    assert graph["removed"] == [1]
    assert pair(items(out)[1]) == (("e", 0), ("e", 1))
    assert items(out)[1]["labels"] == {"p": None, "inc": 1}
