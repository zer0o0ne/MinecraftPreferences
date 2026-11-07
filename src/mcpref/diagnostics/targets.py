"""Producer-time vote, label and graph provenance of (5)-(6)."""

from collections.abc import Mapping, Sequence

from mcpref.diagnostics.config import DiagnosticsConfig
from mcpref.diagnostics.stats import rate
from mcpref.diagnostics.types import MetricRecord, ProducerContext, metric_record

CATEGORIES = ("comparable", "removed", "incomparable", "dropped")


def comparison_context(
    producer: str, model: str, out_dir: str, source: str, comparisons: Sequence[dict]
) -> ProducerContext:
    return ProducerContext(
        producer,
        "train",
        "train",
        model,
        out_dir,
        {},
        (),
        tuple(item["a"]["episode"] for item in comparisons),
        (source,),
    )


def event_id(prefix: str, context: ProducerContext) -> str:
    return f"{prefix}:{int(context.artifact_paths[0].split('/')[-2])}"


def threshold_rate(raw, clusters, threshold, above, cfg):
    values = [
        float(value > threshold) if above else float(value < threshold) for value in raw
    ]
    return rate(values, clusters, cfg)


def on_vote_shard(
    votes: Sequence[dict],
    *,
    weights: Mapping[str, float | None],
    cfg: DiagnosticsConfig,
    context: ProducerContext,
) -> MetricRecord:
    """Raw valid weighted answers, with order-specific strict thresholds."""
    orders = {0: ([], []), 1: ([], [])}
    statuses = {"dropped": 0, "invalid": 0, "no_answer": 0, "INCOMPARABLE": 0}
    for vote in votes:
        if not any(request["label"] in weights for request in vote["requests"]):
            statuses["dropped"] += 1
        for request in vote["requests"]:
            label = request["label"]
            if label in statuses:
                statuses[label] += 1
            weight = weights.get(label)
            if weight is None:
                continue
            order = request["order"]
            orders[order][0].append(float(weight if order == 0 else 1 - weight))
            orders[order][1].append(vote["a"]["episode"])
    metrics = {"status": statuses}
    populations = {}
    for name, threshold, above in (
        ("target.vote_lt_0_2", 0.2, False),
        ("target.vote_gt_0_8", 0.8, True),
    ):
        metrics[name] = {}
        for order, (raw, clusters) in orders.items():
            metrics[name][f"order{order}"] = threshold_rate(
                raw, clusters, threshold, above, cfg
            )
            populations[f"{name}.order{order}"] = list(clusters)
    return metric_record(
        "targets", event_id("vote", context), context, metrics, populations
    )


def on_label_shard(
    labels: Sequence[dict], *, cfg: DiagnosticsConfig, context: ProducerContext
) -> MetricRecord:
    """Corrected preferred-first comparable targets, never pooled with raw votes."""
    comparable = [item for item in labels if not item["labels"]["inc"]]
    values = [item["labels"]["p"] for item in comparable]
    clusters = [item["a"]["episode"] for item in comparable]
    metrics = {
        "target.aggregated_gt_0_8": threshold_rate(values, clusters, 0.8, True, cfg),
        "target.aggregated_lt_0_2_contract": threshold_rate(
            values, clusters, 0.2, False, cfg
        ),
        "comparable": len(values),
        "incomparable": len(labels) - len(values),
    }
    populations = {
        "target.aggregated_gt_0_8": list(clusters),
        "target.aggregated_lt_0_2_contract": list(clusters),
    }
    return metric_record(
        "targets", event_id("label", context), context, metrics, populations
    )


def on_graph(
    comparisons: Sequence[dict],
    *,
    categories: Sequence[str],
    cfg: DiagnosticsConfig,
    context: ProducerContext,
) -> MetricRecord:
    """Graph ratios over aligned original comparisons, including dropped rows."""
    if len(categories) != len(comparisons) or any(
        category not in CATEGORIES for category in categories
    ):
        raise ValueError("categories must align with comparisons")
    clusters = [item["a"]["episode"] for item in comparisons]
    rows = list(zip(categories, clusters))
    graph = [row for row in rows if row[0] in ("comparable", "removed")]
    all_pairs = [row for row in rows if row[0] != "dropped"]

    def metric(population, positive):
        values = [float(category in positive) for category, _ in population]
        aligned = [cluster for _, cluster in population]
        return rate(values, aligned, cfg), aligned

    removed, graph_clusters = metric(graph, {"removed"})
    removed_all, all_clusters = metric(all_pairs, {"removed"})
    semantic, semantic_clusters = metric(all_pairs, {"incomparable"})
    all_inc, inc_clusters = metric(all_pairs, {"removed", "incomparable"})
    metrics = {
        "target.graph_removed": removed,
        "target.graph_removed_all_pairs": removed_all,
        "target.semantic_inc": semantic,
        "target.all_inc": all_inc,
        "categories": {category: categories.count(category) for category in CATEGORIES},
    }
    populations = {
        "target.graph_removed": graph_clusters,
        "target.graph_removed_all_pairs": all_clusters,
        "target.semantic_inc": semantic_clusters,
        "target.all_inc": inc_clusters,
    }
    return metric_record(
        "targets", event_id("graph", context), context, metrics, populations
    )
