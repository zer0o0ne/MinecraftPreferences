"""Inventory-first milestones and right-censored Kaplan–Meier state."""

from mcpref.diagnostics.config import DiagnosticsConfig
from mcpref.diagnostics.observations import EpisodeArtifacts, EpisodeSummary
from mcpref.diagnostics.stats import distribution, rate
from mcpref.diagnostics.types import ProducerContext, metric_record

ITEMS = {
    "stone_pickaxe": ("stone_pickaxe",),
    "iron": ("iron_ingot", "iron_ore", "raw_iron"),
    "iron_pickaxe": ("iron_pickaxe",),
    "diamonds": ("diamond",),
    "obsidian": ("obsidian",),
    "blaze_rod": ("blaze_rod",),
}


def first_times(inventories) -> dict[str, int | None]:
    result = {name: None for name in ITEMS}
    for index, inventory in enumerate(inventories):
        for name, items in ITEMS.items():
            if result[name] is None and any(
                inventory.get(item, 0) > 0 for item in items
            ):
                result[name] = index
    return result


def kaplan_meier(episodes: list[EpisodeSummary], name: str) -> list[dict]:
    events = {}
    for episode in episodes:
        time = episode.first_times[name]
        censored = time is None
        at = episode.censored_at if censored else time
        entry = events.setdefault(at, {"events": 0, "censored": 0})
        entry["censored" if censored else "events"] += 1
    survival, rows = 1.0, []
    for time in sorted(events):
        at_risk = sum(
            (
                episode.censored_at
                if episode.first_times[name] is None
                else episode.first_times[name]
            )
            >= time
            for episode in episodes
        )
        event = events[time]
        if at_risk and event["events"]:
            survival *= 1 - event["events"] / at_risk
        rows.append({"time": time, "survival": survival, "at_risk": at_risk, **event})
    return rows


def on_episode_end(
    episode: EpisodeArtifacts, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(17): first inventory occurrence only; repeats never reset milestone time."""
    current = EpisodeSummary(
        episode.episode,
        first_times(episode.inventories),
        max(len(episode.inventories) - 1, 0),
    )
    episodes = [*episode.previous, current]
    if len({item.episode for item in episodes}) != len(episodes):
        raise ValueError("episode identities must be unique")
    metrics, populations = {}, {}
    for name in ITEMS:
        reached = [item.first_times[name] is not None for item in episodes]
        clusters = [item.episode for item in episodes]
        metrics[f"milestones.coverage_{name}"] = rate(
            [float(value) for value in reached], clusters, cfg
        )
        times = [
            item.first_times[name]
            for item in episodes
            if item.first_times[name] is not None
        ]
        metrics[f"milestones.time_to_first_{name}"] = distribution(times, cfg)
        populations[f"milestones.coverage_{name}"] = clusters
        populations[f"milestones.time_to_first_{name}"] = [
            item.episode for item in episodes if item.first_times[name] is not None
        ]
    joined = {
        "first_times": dict(current.first_times),
        "censored_at": current.censored_at,
        "kaplan_meier": {name: kaplan_meier(episodes, name) for name in ITEMS},
        "order": {
            a: {
                b: current.first_times[a] is not None
                and current.first_times[b] is not None
                and current.first_times[a] < current.first_times[b]
                for b in ITEMS
            }
            for a in ITEMS
        },
    }
    return metric_record(
        "milestones", f"episode:{episode.episode}", context, metrics, populations
    ) | {"joined": joined}
