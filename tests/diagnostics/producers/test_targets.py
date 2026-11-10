"""Equations (2)–(6): target populations, exact rates and online intervals."""

import math
import statistics

import numpy as np
import pytest

from diagnostics.core.world import comparison, config, context, label, vote
from mcpref.diagnostics.producers.targets import on_graph, on_label_shard, on_vote_shard


def test_vote_strict_thresholds_exclusions_status_and_order_populations():
    weights = {"LOW0": 0.0, "LOW1": 0.199, "EDGE_L": 0.2,
               "EDGE_H": 0.8, "HIGH0": 0.801, "HIGH1": 1.0, "INCOMPARABLE": None}  # fmt: skip
    tokens = list(weights)
    votes = [vote("e0", tokens + ["invalid", None])]
    record = on_vote_shard(
        votes,
        weights=weights,
        cfg=config(),
        context=context("pair_votes/shards/000007/answers.json", "vlm-labels"),
    )
    assert record["id"] == "vote:7" and record["joined"] == {}
    for order in (0, 1):
        low = record["metrics"]["target.vote_lt_0_2"][f"order{order}"]
        high = record["metrics"]["target.vote_gt_0_8"][f"order{order}"]
        low_numerator = 2 if order == 0 else 3  # 1 - 0.8 is below 0.2 in float64
        assert (low["n"], low["numerator"], low["value"]) == (
            6,
            low_numerator,
            low_numerator / 6,
        )
        assert (high["n"], high["numerator"], high["value"]) == (6, 2, 1 / 3)
        for name in ("target.vote_lt_0_2", "target.vote_gt_0_8"):
            assert record["populations"][f"{name}.order{order}"] == ["e0"] * 6
    assert record["metrics"]["status"] == {
        "dropped": 0,
        "invalid": 2,
        "no_answer": 2,
        "INCOMPARABLE": 2,
    }


def test_label_uses_corrected_comparable_population_and_contract_check():
    labels = [label("e0", 0.9, 0), label("e1", 0.1, 0), label("e2", None, 1)]
    record = on_label_shard(labels, cfg=config(), context=context())
    high = record["metrics"]["target.aggregated_gt_0_8"]
    low = record["metrics"]["target.aggregated_lt_0_2_contract"]
    assert (high["n"], high["numerator"], high["value"]) == (2, 1, 0.5)
    assert (low["n"], low["numerator"], low["value"]) == (2, 1, 0.5)
    assert record["populations"]["target.aggregated_gt_0_8"] == ["e0", "e1"]
    assert record["metrics"]["comparable"] == 2
    assert record["metrics"]["incomparable"] == 1


def test_graph_hand_populations_and_dropped_outside_denominator():
    comparisons = [
        comparison("e0"),
        comparison("e1"),
        comparison("e2"),
        comparison("e3"),
    ]
    categories = ["comparable", "removed", "incomparable", "dropped"]
    record = on_graph(
        comparisons,
        categories=categories,
        cfg=config(),
        context=context("pair_labels/shards/000003/counts.json", "acyclic-graph"),
    )
    expected = {
        "target.graph_removed": (2, 1, 0.5),
        "target.graph_removed_all_pairs": (3, 1, 1 / 3),
        "target.semantic_inc": (3, 1, 1 / 3),
        "target.all_inc": (3, 2, 2 / 3),
    }
    for name, (n, numerator, value) in expected.items():
        metric = record["metrics"][name]
        assert (metric["n"], metric["numerator"]) == (n, numerator)
        assert metric["value"] == pytest.approx(value)
    assert record["populations"]["target.graph_removed"] == ["e0", "e1"]
    assert record["populations"]["target.all_inc"] == ["e0", "e1", "e2"]
    assert record["metrics"]["categories"] == dict.fromkeys(
        ("comparable", "removed", "incomparable", "dropped"), 1
    )


def test_graph_exact_current_train_counts_and_four_distinct_rates():
    c, f, semantic = 81430, 23669, 264
    categories = (
        ["comparable"] * c + ["removed"] * f + ["incomparable"] * semantic + ["dropped"]
    )
    row = comparison("e0")
    record = on_graph(
        [row] * len(categories),
        categories=categories,
        cfg=config(bootstrap_samples=1),
        context=context(
            "pair_labels/shards/000000/counts.json", "acyclic-graph", clusters=("e0",)
        ),
    )
    expected = {
        "target.graph_removed": (c + f, f, 0.22520670986403296),
        "target.graph_removed_all_pairs": (c + f + semantic, f, 0.22464242665831458),
        "target.semantic_inc": (c + f + semantic, semantic, 0.002505623416189744),
        "target.all_inc": (c + f + semantic, f + semantic, 0.22714805007450434),
    }
    for name, (n, numerator, value) in expected.items():
        metric = record["metrics"][name]
        assert (metric["n"], metric["numerator"]) == (n, numerator)
        assert metric["value"] == pytest.approx(value, rel=1e-15)
    assert record["metrics"]["categories"]["dropped"] == 1


def wilson(x: int, n: int, level: float) -> dict:
    z = statistics.NormalDist().inv_cdf((1 + level) / 2)
    p = x / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    delta = z / (1 + z * z / n) * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return {"low": max(0, center - delta), "high": min(1, center + delta)}


def test_wilson_and_real_cluster_bootstrap_match_independent_oracle():
    cfg = config(bootstrap_samples=16, min_cluster_count=2, quantiles=[0.05, 0.95])
    rows = [comparison("e0"), comparison("e0"), comparison("e1")]
    record = on_graph(
        rows,
        categories=["comparable", "removed", "incomparable"],
        cfg=cfg,
        context=context("pair_labels/shards/000003/counts.json", "acyclic-graph"),
    )
    metric = record["metrics"]["target.all_inc"]
    assert metric["wilson"] == pytest.approx(wilson(2, 3, 0.95))
    rng = np.random.Generator(np.random.PCG64([0, 5]))
    replicates = []
    draws = []
    for _ in range(16):
        drawn = rng.choice(2, size=2, replace=True, p=None, axis=0, shuffle=True)
        draws.append(drawn)
        numerator = sum((1, 1)[cluster] for cluster in drawn)
        denominator = sum((2, 1)[cluster] for cluster in drawn)
        replicates.append(float(numerator / denominator))
    assert any(len(set(drawn)) < 2 for drawn in draws)
    from diagnostics.core.world import quantile

    assert metric["bootstrap"] == pytest.approx(
        {"low": quantile(replicates, 0.05), "high": quantile(replicates, 0.95)}
    )
    assert metric["insufficient_clusters"] is False


def test_insufficient_clusters_preserve_raw_rate_and_null_bounds():
    record = on_graph(
        [comparison("e0")],
        categories=["removed"],
        cfg=config(min_cluster_count=2),
        context=context(
            "pair_labels/shards/000003/counts.json", "acyclic-graph", clusters=("e0",)
        ),
    )
    metric = record["metrics"]["target.graph_removed"]
    assert metric["value"] == 1 and metric["bootstrap"] == {"low": None, "high": None}
    assert metric["insufficient_clusters"] is True


@pytest.mark.parametrize("categories", [["comparable"], ["unknown", "dropped"]])
def test_graph_rejects_misaligned_or_unknown_categories(categories):
    with pytest.raises(ValueError):
        on_graph(
            [comparison("e0"), comparison("e1")],
            categories=categories,
            cfg=config(),
            context=context("pair_labels/shards/000003/counts.json", "acyclic-graph"),
        )


def test_hooks_require_keyword_only_cfg_and_context():
    with pytest.raises(TypeError):
        on_vote_shard([], weights={})
    with pytest.raises(TypeError):
        on_label_shard([], cfg=config())
    with pytest.raises(TypeError):
        on_graph([], categories=[], context=context())
