"""Memory-event aggregation from existing raw work and pending History."""

import json

from mcpref import REPO
from mcpref.diagnostics.memory import on_memory_event
from mcpref.diagnostics.observations import MemoryEvent
from mcpref.diagnostics.runtime import DiagnosticRun
from mcpref.diagnostics.types import ProducerContext
from mcpref.rl_pairs.sample import read_work
from mcpref.shards.store import complete_shards, shard_path


def register(diagnostics: DiagnosticRun, cycle: int) -> None:
    watermark = diagnostics.state["channels"].get("memory", {}).get("watermark", cycle)
    diagnostics.register("memory", watermark=watermark, next=cycle)


def stage(
    diagnostics: DiagnosticRun, cfg, cycle: int, real_states: int, history
) -> None:
    work = REPO / cfg.out_dir / f"{cycle:06d}" / "work"
    totals = {
        "temporal": 0,
        "nearest": 0,
        "beam": 0,
        "empty_candidates": 0,
        "repeated": 0,
        "occurrences": 0,
        "comparisons": 0,
    }
    clusters = []
    for k in range(complete_shards(work)):
        result = read_work(shard_path(work, k))
        observed = result["diagnostics"]
        for key in ("temporal", "nearest", "beam", "empty_candidates", "repeated"):
            totals[key] += observed[key]
        totals["occurrences"] += len(result["alternatives"])
        totals["comparisons"] += len(result["comparisons"])
        clusters.extend(result["seed"]["episode"] for _ in result["comparisons"])
    vertices = list(history._vertices())
    cleanup = REPO / cfg.out_dir / f"{cycle:06d}" / "cleanup.json"
    removed = json.loads(cleanup.read_text())["removed"] if cleanup.exists() else []
    episode = f"{cfg.out_dir}#{cycle:06d}"
    covered = tuple(sorted({vertex.ref.episode for vertex in vertices}))
    episodes = tuple(sorted({episode, *covered}))
    event = MemoryEvent(
        "publication",
        real_states,
        complete_shards(work),
        totals["temporal"],
        totals["nearest"],
        totals["beam"],
        totals["occurrences"],
        totals["comparisons"],
        totals["repeated"],
        len(vertices),
        cfg.capacity,
        len(removed),
        sum(vertex.unused for vertex in vertices),
        totals["empty_candidates"],
        episodes,
        covered,
    )
    source = f"{cfg.out_dir}/{cycle:06d}/run.json"
    context = ProducerContext(
        f"memory:{cycle:06d}",
        "train",
        "train",
        f"{cfg.out_dir}/{cycle:06d}",
        source,
        {},
        (cycle,),
        tuple(clusters),
        (source,),
    )
    record = on_memory_event(event, cfg=diagnostics.cfg, context=context)
    diagnostics.stage("memory", cycle, record["id"], cycle + 1, [source], record)


def accept(diagnostics: DiagnosticRun, cfg, cycle: int) -> None:
    run_path = REPO / cfg.out_dir / f"{cycle:06d}" / "run.json"
    if not json.loads(run_path.read_text()).get("complete"):
        return
    diagnostics.recover("memory")
    diagnostics.finish(
        "memory",
        source_complete=True,
        done=diagnostics.state["channels"]["memory"]["next"] >= cycle + 1,
    )
