"""Stage-33a vote, label and graph producer handoffs."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from mcpref.diagnostics.runtime import DiagnosticRun
from mcpref.diagnostics.targets import (
    comparison_context,
    on_graph,
    on_label_shard,
    on_vote_shard,
)
from mcpref.runs import files
from mcpref.shards.store import name, shard_path, write_shard

if TYPE_CHECKING:
    from mcpref.preferences.acyclic import AcyclicConfig
    from mcpref.preferences.labels import VlmLabelsConfig


def ordered(tally: Counter, cfg: "VlmLabelsConfig") -> dict:
    keys = [entry["token"] for entry in cfg.answers] + ["invalid", "no_answer"]
    return {key: tally[key] for key in keys if key in tally}


def summed(labels: Path, n: int) -> tuple[int, Counter, int]:
    comparisons, tally, end = 0, Counter(), 0
    for k in range(n):
        counts = json.loads((shard_path(labels, k) / "counts.json").read_bytes())
        comparisons, end = comparisons + counts["comparisons"], counts["end"]
        tally.update(counts["labels"])
    return comparisons, tally, end


def write_vote(
    labels: Path,
    k: int,
    entries: list,
    end: int,
    cfg: "VlmLabelsConfig",
    *,
    diagnostics: DiagnosticRun,
    weights: Mapping[str, float | None],
) -> None:
    tally = Counter(
        request["label"] for entry in entries for request in entry["requests"]
    )
    counts = {"comparisons": len(entries), "labels": ordered(tally, cfg), "end": end}
    source = f"{cfg.out_dir}/shards/{name(k)}/answers.json"
    context = comparison_context("vlm-labels", "vlm", cfg.out_dir, source, entries)
    record = on_vote_shard(
        entries, weights=weights, cfg=diagnostics.cfg, context=context
    )
    diagnostics.stage("vote", k, f"vote:{k}", k + 1, [source], record)
    with write_shard(labels, k) as staged:
        (staged / "answers.json").write_bytes(files.json_bytes(entries))
        (staged / "counts.json").write_bytes(files.json_bytes(counts))
    diagnostics.accept("vote", f"vote:{k}")


def stage_acyclic(
    k: int,
    comparisons: list,
    pairs: list,
    categories: Sequence[str],
    cfg: "AcyclicConfig",
    diagnostics: DiagnosticRun,
) -> None:
    pair_source = f"{cfg.out_dir}/shards/{name(k)}/pairs.json"
    count_source = f"{cfg.out_dir}/shards/{name(k)}/counts.json"
    label_context = comparison_context(
        "acyclic-labels", "acyclic", cfg.out_dir, pair_source, comparisons
    )
    graph_context = comparison_context(
        "acyclic-graph", "acyclic", cfg.out_dir, count_source, comparisons
    )
    label_record = on_label_shard(pairs, cfg=diagnostics.cfg, context=label_context)
    graph_record = on_graph(
        comparisons, categories=categories, cfg=diagnostics.cfg, context=graph_context
    )
    diagnostics.stage("label", k, f"label:{k}", k + 1, [pair_source], label_record)
    diagnostics.stage("graph", k, f"graph:{k}", k + 1, [count_source], graph_record)


def accept_acyclic(diagnostics: DiagnosticRun, k: int) -> None:
    diagnostics.accept("label", f"label:{k}")
    diagnostics.accept("graph", f"graph:{k}")
