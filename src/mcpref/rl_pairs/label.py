"""Eqs. (8)-(9): temporal thresholds and current-cycle acyclic correction."""

import json
import math
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

from mcpref.graphs.feedback import removed_arcs
from mcpref.rl_pairs.config import RLPreferencePairsConfig
from mcpref.rl_pairs.files import shard_items, write_json_shard
from mcpref.rl_pairs.sample import read_work
from mcpref.rl_pairs.types import (
    CurrentComparison,
    StateRef,
    current_comparison,
    ref_dict,
)
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards, shard_path


def threshold(a: StateRef, b: StateRef, cfg: RLPreferencePairsConfig) -> float:
    """Eq. (8): the close-pair sigmoid threshold or cross-episode kappa*tau_0."""
    if a.episode != b.episode:
        return cfg.threshold_scale * cfg.base_threshold
    distance = abs(a.index - b.index)
    logit = math.log(cfg.base_threshold / (1 - cfg.base_threshold))
    sigmoid = 1 / (1 + math.exp(-(logit - cfg.temporal_penalty / distance)))
    return cfg.threshold_scale * sigmoid


def graph_entry(
    c: CurrentComparison,
    cfg: RLPreferencePairsConfig,
    vertex: Callable[[StateRef], int],
    arcs: list,
    costs: list,
) -> dict:
    """One raw comparison as an initially incomparable entry or preferred-first arc."""
    entry = {"a": ref_dict(c.a), "b": ref_dict(c.b), "w": None, "arc": None}
    if c.inc > threshold(c.a, c.b, cfg):
        return entry
    w = max(c.p, 1 - c.p)
    a, b = (c.a, c.b) if c.p >= 0.5 else (c.b, c.a)
    arc = len(arcs)
    arcs.append([vertex(a), vertex(b)])
    costs.append(Fraction(str(w)) - Fraction(1, 2))
    return {"a": ref_dict(a), "b": ref_dict(b), "w": w, "arc": arc}


def build_graph(work_dir: Path, out_dir: Path, cfg: RLPreferencePairsConfig) -> dict:
    """Build the single current-comparison graph work item in comparison order."""
    vertices: list[StateRef] = []
    ids: dict[StateRef, int] = {}
    arcs = []
    costs = []

    def vertex(ref: StateRef) -> int:
        if ref not in ids:
            ids[ref] = len(vertices)
            vertices.append(ref)
        return ids[ref]

    comparisons = []
    for k in range(complete_shards(work_dir)):
        result = read_work(shard_path(work_dir, k))
        for value in result["comparisons"]:
            c = current_comparison(value)
            comparisons.append(graph_entry(c, cfg, vertex, arcs, costs))
    removed = removed_arcs(len(vertices), arcs, costs)
    graph = {
        "vertices": [ref_dict(ref) for ref in vertices],
        "arcs": arcs,
        "costs": [str(cost) for cost in costs],
        "removed": removed,
        "comparisons": comparisons,
    }
    write_json(out_dir / "graph.json", graph)
    return graph


def correct_remaining(
    comparisons: list,
    removed: set,
    out_dir: Path,
    cfg: RLPreferencePairsConfig,
    shards: int,
    done: int,
) -> None:
    """Write the unfinished corrected items with per-pair progress."""
    with tqdm(
        total=len(comparisons), initial=done, desc="rl pair correction", unit="pair"
    ) as bar:
        items = []
        for entry in comparisons[done:]:
            if entry["w"] is None or entry["arc"] in removed:
                labels = {"p": None, "inc": 1}
            else:
                labels = {"p": entry["w"], "inc": 0}
            items.append({"a": entry["a"], "b": entry["b"], "labels": labels})
            if len(items) == cfg.items_per_shard:
                write_json_shard(out_dir / "shards", shards, "items.json", items)
                shards += 1
                items = []
            bar.update(1)
        if items:
            write_json_shard(out_dir / "shards", shards, "items.json", items)


def correct(work_dir: Path, out_dir: Path, cfg: RLPreferencePairsConfig) -> int:
    """Eq. (9), resumed after graph creation and complete Q_c shards."""
    graph_path = out_dir / "graph.json"
    graph = (
        json.loads(graph_path.read_text())
        if graph_path.exists()
        else build_graph(work_dir, out_dir, cfg)
    )
    comparisons = graph["comparisons"]
    total = len(comparisons)
    shards = complete_shards(out_dir / "shards")
    done = shard_items(out_dir / "shards", "items.json", shards, cfg.items_per_shard)
    if total == 0:
        print("rl pair correction: nothing to do", file=sys.stderr)
        return 0
    if done == total:
        print("rl pair correction: already done", file=sys.stderr)
        return total
    correct_remaining(comparisons, set(graph["removed"]), out_dir, cfg, shards, done)
    return total
