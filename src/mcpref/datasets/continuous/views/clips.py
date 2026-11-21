"""Clip metadata references existing grid observations without copying pixels."""

from mcpref.datasets.clips.sample import gui_flag

FORMAT = "mcpref-grid-clip-v1"


def clip_info(mapped: dict, timeline: dict, delta: int, dataset: str) -> dict:
    ordinals = [p + mapped["shift"] for p in mapped["original_kept_ordinals"]]
    indices = [p // delta for p in ordinals]
    raw = [timeline["kappa"][p] for p in ordinals]
    intervals = [timeline["intervals"][j] for j in indices[:-1]]
    flags = [gui_flag(timeline["rows"], t) for t in raw]
    return {
        "format": FORMAT,
        "dataset": dataset,
        "id": mapped["canonical_history_id"],
        "relpath": timeline["id"],
        "fragment_id": timeline["id"],
        "observation_ids": mapped["observations"],
        "frames": [timeline["locations"][p] for p in ordinals],
        "states": list(range(len(raw))),
        "actions": [i["actions"] for i in intervals],
        "n": [i["n"] for i in intervals],
        "gui": flags,
        "classes": [
            timeline["classes"][t] if flag else None for t, flag in zip(raw, flags)
        ],
        "pitch": [timeline["rows"][t]["pitch"] for t in raw],
        "classical_valid": all(i["valid"] for i in intervals),
        "classical_reasons": sorted(
            {reason for i in intervals if not i["valid"] for reason in i["reasons"]}
        ),
    }
