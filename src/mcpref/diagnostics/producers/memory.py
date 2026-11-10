"""Memory/data-efficiency observations with explicit source populations."""

from mcpref.diagnostics.core.config import DiagnosticsConfig
from mcpref.diagnostics.core.observations import MemoryEvent
from mcpref.diagnostics.core.stats import distribution, rate
from mcpref.diagnostics.core.types import ProducerContext, metric_record

KINDS = (
    "seed",
    "temporal",
    "nearest",
    "beam",
    "comparison",
    "eviction",
    "cleanup",
    "publication",
)


def _ratio(numerator: int, denominator: int, cfg: DiagnosticsConfig, reason: str):
    value = numerator / denominator if denominator else None
    return distribution(
        [] if value is None else [value],
        cfg,
        missing=int(value is None),
        reason=reason if value is None else None,
    )


def on_memory_event(
    event: MemoryEvent, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(16): counts and ratios from one existing memory event."""
    if event.kind not in KINDS:
        raise ValueError(f"unknown memory event kind: {event.kind}")
    metrics = {
        "memory.seeds": event.seeds,
        "memory.temporal": event.temporal,
        "memory.nearest": event.nearest,
        "memory.beam": event.beam,
        "memory.evictions": event.evictions,
        "memory.unused": event.unused,
        "memory.empty_candidates": event.empty_candidates,
        "memory.dedup_ratio": _ratio(
            event.candidate_occurrences - event.unique_comparisons,
            event.candidate_occurrences,
            cfg,
            "zero_candidate_occurrences",
        ),
        "memory.capacity": _ratio(event.vertices, event.capacity, cfg, "zero_capacity"),
    }
    for name, count in (
        ("seeds", event.seeds),
        ("temporal", event.temporal),
        ("nearest", event.nearest),
        ("beam", event.beam),
    ):
        metrics[f"memory.{name}_per_real_state"] = _ratio(
            count, event.real_states, cfg, "zero_real_states"
        )
        metrics[f"memory.{name}_per_seed"] = _ratio(
            count, event.seeds, cfg, "zero_seeds"
        )
    metrics["memory.empty_candidates_per_real_state"] = _ratio(
        event.empty_candidates, event.real_states, cfg, "zero_real_states"
    )
    repeated = [1.0] * event.repeated_comparisons + [0.0] * (
        event.unique_comparisons - event.repeated_comparisons
    )
    if len(repeated) != event.unique_comparisons or event.repeated_comparisons < 0:
        raise ValueError("repeated comparisons must be a subset of unique comparisons")
    metrics["memory.repeated_comparison"] = rate(
        repeated, list(context.cluster_ids), cfg
    )
    episodes = list(dict.fromkeys(event.episodes))
    covered = set(event.covered_episodes)
    if not covered <= set(episodes):
        raise ValueError("covered episodes must be a subset of recorded episodes")
    metrics["memory.episode_coverage"] = rate(
        [float(episode in covered) for episode in episodes], episodes, cfg
    )
    populations = {
        name: list(context.cluster_ids)
        for name, value in metrics.items()
        if isinstance(value, dict)
    }
    populations["memory.episode_coverage"] = episodes
    step = context.steps[0] if context.steps else 0
    return metric_record(
        "rl", f"memory:{step}:{event.kind}", context, metrics, populations
    )
