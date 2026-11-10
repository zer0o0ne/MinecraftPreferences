"""Memory source populations and independent milestone/Kaplan–Meier oracles."""

from mcpref.diagnostics.memory import on_memory_event
from mcpref.diagnostics.milestones import on_episode_end
from mcpref.diagnostics.observations import (
    EpisodeArtifacts,
    EpisodeSummary,
    MemoryEvent,
)

from .world import config, context


def test_memory_ratios_repeated_comparisons_and_episode_coverage():
    event = MemoryEvent(
        "comparison", 10, 2, 3, 4, 5, 8, 3, 1, 7, 10, 2, 1, 1, ("e0", "e1"), ("e0",)
    )
    record = on_memory_event(
        event,
        cfg=config(),
        context=context(
            "rl_pairs/000000/work/result.json",
            "memory",
            steps=(4,),
            clusters=("e0", "e0", "e1"),
        ),
    )
    metrics = record["metrics"]
    assert metrics["memory.seeds"] == 2
    assert metrics["memory.temporal_per_real_state"]["mean"] == 0.3
    assert metrics["memory.beam_per_seed"]["mean"] == 2.5
    assert metrics["memory.dedup_ratio"]["mean"] == 5 / 8
    assert metrics["memory.repeated_comparison"]["value"] == 1 / 3
    assert metrics["memory.capacity"]["mean"] == 0.7
    assert metrics["memory.episode_coverage"]["value"] == 0.5


def test_milestones_first_occurrence_repeat_pickup_censoring_and_survival():
    previous = EpisodeSummary(
        "old",
        {
            "stone_pickaxe": 1,
            "iron": None,
            "iron_pickaxe": None,
            "diamonds": None,
            "obsidian": None,
            "blaze_rod": None,
        },
        3,
    )
    episode = EpisodeArtifacts(
        "new",
        [
            {},
            {"iron_ore": 1},
            {"stone_pickaxe": 1},
            {"stone_pickaxe": 2},
        ],
        [previous],
    )
    record = on_episode_end(
        episode,
        cfg=config(),
        context=context(
            "sampling/shards/000000", "milestones", steps=(3,), clusters=("new",)
        ),
    )
    metrics = record["metrics"]
    assert record["joined"]["first_times"]["stone_pickaxe"] == 2
    assert record["joined"]["first_times"]["iron"] == 1
    assert metrics["milestones.coverage_stone_pickaxe"]["value"] == 1.0
    assert metrics["milestones.time_to_first_stone_pickaxe"]["median"] == 1.5
    curve = record["joined"]["kaplan_meier"]["stone_pickaxe"]
    assert [(row["time"], row["survival"]) for row in curve] == [(1, 0.5), (2, 0.0)]
    assert metrics["milestones.coverage_blaze_rod"]["value"] == 0.0
