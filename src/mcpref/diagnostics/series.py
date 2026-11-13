"""Extract plot series while preserving producer populations and source snapshots."""

import hashlib
import math
from collections.abc import Iterable, Sequence

from mcpref.diagnostics.core.artifacts import (
    GROUP_PANELS,
    PLOT_REGISTRY,
    group_registry,
    pretrain_registry,
)

ALIASES = {
    "preference.loss": "P",
    "policy.loss": "pi",
    "perception.loss": "phi",
    "factual_shift.loss": "shift",
    "metra.loss": "metra",
}


def at(value: dict, path: str):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def source(record: dict, key: str):
    key = ALIASES.get(key, key)
    if key.startswith("joined."):
        return at(record, key)
    if key in record["metrics"] and scalar(record["metrics"][key]) is not None:
        return record["metrics"][key]
    joined = record["joined"]
    if key in joined.get("outputs", {}):
        return joined["outputs"][key]
    if key in joined.get("condition", {}).get("losses", {}):
        return joined["condition"]["losses"][key]
    if not key.startswith("preference."):
        return record["metrics"].get(key)
    path = key.removeprefix("preference.")
    if "condition" in joined:
        return at(joined["condition"]["preference"], path)
    window = joined.get("window_preference")
    return at(window, path) if window is not None else None


def scalar(value):
    if isinstance(value, dict):
        result = value.get("value")
        return value.get("mean") if result is None else result
    return value if isinstance(value, (int, float)) else None


def xy(record: dict, key: str, axis: str):
    value = scalar(source(record, key))
    if value is None:
        return None
    if axis == "index":
        return record["index"], value
    steps = record["context"]["steps"]
    if len(steps) != 1:
        raise RuntimeError("diagnostic plots require exactly one source step")
    return steps[0], value


def identity(record: dict) -> tuple[str, ...]:
    context = record["context"]
    return (
        record["kind"],
        *(context[key] for key in ("producer", "source", "split", "condition")),
    )


def curve_points(record: dict, entry: dict, name: str) -> list[tuple[float, float]]:
    axis = entry["x"]
    joined = record["joined"]
    if axis == "bin":
        rows = joined.get("policy", joined).get("reliability", [])
        return [
            (row["bin"], row[name] if row[name] is not None else math.nan)
            for row in rows
        ]
    if axis == "matched_mass":
        return [
            (row["mass"], float(row[name]))
            for row in joined.get("matched_reliability", [])
        ]
    if axis.startswith("km:"):
        rows = joined.get("kaplan_meier", {}).get(axis.removeprefix("km:"), [])
        return [(row["time"], row[name]) for row in rows]
    point = xy(record, name, axis)
    if point is not None:
        return [point]
    if axis == "step" and len(record["context"]["steps"]) == 1:
        return [(record["context"]["steps"][0], math.nan)]
    return [(record["index"], math.nan)]


def has_snapshot(record: dict, axis: str) -> bool:
    joined = record["joined"]
    if axis == "bin":
        return "reliability" in joined.get("policy", joined)
    if axis == "matched_mass":
        return "matched_reliability" in joined
    return axis.removeprefix("km:") in joined.get("kaplan_meier", {})


def candidates(record: dict) -> list[dict]:
    facet = [(record["context"]["split"], record["context"]["condition"])]
    entries = list(PLOT_REGISTRY)
    if record["kind"] == "pretrain":
        entries += pretrain_registry(facet)
    extra = sorted(key for key in record["metrics"] if key.startswith("update.group."))
    for group in GROUP_PANELS:
        entries += group_registry(
            group, facet, extra if group == "update-health" else ()
        )
    return entries


def registries(records: Iterable[dict]) -> list[dict]:
    result = {}
    for record in records:
        key = identity(record)
        suffix = hashlib.sha256("\0".join(key).encode()).hexdigest()[:12]
        for entry in candidates(record):
            axis = entry["x"]
            available = any(
                source(record, name) is not None for name in entry["series"]
            )
            if axis not in ("step", "index"):
                available = any(
                    curve_points(record, entry, name) for name in entry["series"]
                )
            if not available:
                continue
            filename = entry["file"].removesuffix(".png") + f"__{suffix}.png"
            if filename in result:
                result[filename]["series"] = sorted(
                    set(result[filename]["series"]) | set(entry["series"])
                )
                continue
            context = record["context"]
            title = (
                f"{entry['group']} | {context['producer']} | "
                f"{context['split']}/{context['condition']}\n{context['source']}"
            )
            result[filename] = entry | {
                "file": filename,
                "identity": key,
                "title": title,
            }
    pages = []
    for entry in result.values():
        for offset in range(0, len(entry["series"]), 6):
            page = entry | {"series": entry["series"][offset : offset + 6]}
            if offset:
                page["file"] = (
                    entry["file"].removesuffix(".png") + f"__page-{offset // 6 + 1}.png"
                )
            pages.append(page)
    return pages


def collect(records: Iterable[dict], entries: Sequence[dict]) -> dict:
    points = {
        entry["file"]: {name: [] for name in entry["series"]} for entry in entries
    }
    by_identity = {}
    for entry in entries:
        by_identity.setdefault(entry["identity"], []).append(entry)
    for record in records:
        for entry in by_identity.get(identity(record), []):
            snapshot = entry["x"] not in ("step", "index")
            if snapshot and not has_snapshot(record, entry["x"]):
                continue
            for name in entry["series"]:
                values = curve_points(record, entry, name)
                target = points[entry["file"]][name]
                if snapshot:
                    target.clear()
                target.extend(values)
            if snapshot:
                entry["snapshot"] = (
                    f"step {record['context']['steps']}, record {record['index']}"
                )
    for series in points.values():
        for values in series.values():
            values.sort(key=lambda point: point[0])
    return points


def axis_label(entry: dict) -> str:
    axis = entry["x"]
    if axis == "index":
        return "Accepted record index"
    if axis == "bin":
        return "Confidence bin"
    if axis == "matched_mass":
        return "Matched probability mass"
    if axis.startswith("km:"):
        return "Time to milestone (real actions)"
    if entry["group"] == "memory-efficiency":
        return "Cycle"
    if entry["group"] in ("policy-mcts", "mcts-outcomes", "milestones"):
        return "Real action"
    return "Optimizer step (local to source)"
