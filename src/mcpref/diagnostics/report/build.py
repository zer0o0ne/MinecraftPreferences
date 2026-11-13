"""Route existing coordinates into dashboards without averaging populations."""

import math
from pathlib import Path

from mcpref.config import REPO
from mcpref.diagnostics.report.catalog import definitions
from mcpref.diagnostics.report.populations import cycle_view
from mcpref.diagnostics.report.routing import (
    axis_spec,
    canonical,
    group_number,
    is_reference,
    readable,
)
from mcpref.diagnostics.series import axis_label


def population(entry: dict, metric: str) -> tuple:
    if "identity" in entry:
        _kind, producer, source, split, condition = entry["identity"]
        mode = "Reference" if is_reference(entry) else "Steps"
        if producer == "pretrain" and metric.startswith("preference."):
            mode = "Windows"
        # Distinct producers can have different observation populations.
        origin = (
            producer
            if producer not in ("pretrain", "pretrain-reference")
            else "Pretrain"
        )
        label = f"{origin} / {mode.lower()} / {split}"
    else:
        source = entry["source"]
        mode = entry["group"].rsplit("-", 1)[-1].title()
        split, condition = "train", "joint"
        label = entry["population"]
        origin = label.split(" / ")[0]
        if mode == "Reference":
            path = Path(source)
            source, split, condition = str(path.parents[2]), path.parent.name, path.name
            label = f"{origin} / reference / {split}"
        elif "retrain" in label:
            condition = "retrain"
    root = str(
        (REPO / source).resolve() if "identity" in entry else Path(source).resolve()
    )
    short_source = "/".join(Path(source).parts[-2:])
    return (root, origin, mode, split), condition, f"{label}\n{short_source}", source


def curve_label(metric: str, entry: dict, condition: str) -> str:
    if entry["x"].startswith("km:"):
        return readable(entry["x"].removeprefix("km:"))
    label = readable(canonical(metric))
    if condition not in ("joint", "retrain") or is_reference(entry):
        label += f" · {condition}"
    if metric in ("policy.prior_ce", "ppo.eligible") or (
        "topk" in metric and entry.get("identity", ("", ""))[1] == "pretrain"
    ):
        label += " [producer issue]"
    return label


def compact_conditions(curves: list[dict]) -> list[dict]:
    """Draw identical reference-condition curves once, explicitly naming all aliases."""
    compact = {}
    for curve in curves:
        coordinates = tuple(
            (x, y if math.isfinite(y) else None) for x, y in curve["points"]
        )
        key = (canonical(curve["metric"]), curve["split"], curve["mode"], coordinates)
        if key in compact:
            compact[key]["conditions"].append(curve["condition"])
        else:
            compact[key] = curve | {"conditions": [curve["condition"]]}
    for curve in compact.values():
        if curve["mode"] == "Reference":
            conditions = curve["conditions"]
            suffix = (
                "all conditions"
                if set(conditions) == {"joint", "initial", "P_only", "phi_only"}
                else ", ".join(conditions)
            )
            curve["label"] = (
                f"{readable(canonical(curve['metric']))} · {curve['split']} · {suffix}"
            )
    return list(compact.values())


def add_curve(
    groups: dict, metadata: list, entry: dict, metric: str, values: list
) -> None:
    number = group_number(entry, metric)
    target = metadata[number - 1]
    target["series"].append(metric)
    if "identity" in entry:
        target["accepted_series"].append(metric)
    pop, condition, label, source = population(entry, metric)
    target["sources"].append(source)
    if not any(math.isfinite(y) for _, y in values):
        target["missing"] += 1
        return
    title, unit = axis_spec(metric, entry, number)
    root, origin, mode, split = pop
    if mode == "Reference":
        pop = (root, origin, mode)
        label = f"{origin} / fixed reference\n{'/'.join(Path(source).parts[-2:])}"
    elif mode in ("Steps", "Windows"):
        pop = (root, origin, "Training", split)
        label = f"{origin} / {split}\n{'/'.join(Path(source).parts[-2:])}"
    axis = entry["x"]
    x_label = entry.get("x_label", axis_label(entry))
    cycle = cycle_view(entry)
    cycle_label = ""
    if cycle is not None:
        run_root, role, phase, cycle_index = cycle
        pop = (run_root, role, phase, split)
        label = f"RL / {role} / {phase}\n{'/'.join(Path(run_root).parts[-2:])}"
        cycle_label = "" if role == "memory" else f"cycle {cycle_index}"
        x_label = "Cycle" if role == "memory" else "Optimizer step within cycle"
    if axis.startswith("km:"):
        axis = "km"
    snapshot = entry.get("snapshot")
    key = (number, pop, title, unit, axis, snapshot)
    panel = groups.setdefault(
        key,
        {
            "title": title,
            "population": label,
            "unit": unit,
            "x_label": x_label,
            "snapshot": snapshot,
            "step": axis == "km",
            "curves": {},
        },
    )
    curve_key = (
        canonical(metric) if axis != "km" else entry["x"],
        condition,
        split,
        mode,
        cycle_label,
    )
    curve = panel["curves"].setdefault(
        curve_key,
        {
            "metric": metric,
            "label": curve_label(metric, entry, condition),
            "points": {},
            "condition": condition,
            "split": split,
            "mode": mode,
            "cycle": cycle_label,
        },
    )
    if mode == "Windows":
        curve["label"] = curve_label(metric, entry, condition) + " · window"
    if cycle_label:
        curve["label"] = curve_label(metric, entry, condition) + f" · {cycle_label}"
    for x, y in values:
        # Duplicate registry aliases refer to the same saved coordinate, never a mean.
        if math.isfinite(y) or x not in curve["points"]:
            curve["points"][x] = y


def dashboards(entries: list[dict], points: dict) -> tuple[list[dict], dict]:
    metadata, groups = definitions(), {}
    for entry in entries:
        for metric, values in points[entry["file"]].items():
            add_curve(groups, metadata, entry, metric, values)
    panels = {entry["file"]: [] for entry in metadata}
    for (number, *_), panel in groups.items():
        curves = list(panel["curves"].values())
        for curve in curves:
            curve["points"] = sorted(curve["points"].items())
        if curves[0]["mode"] == "Reference":
            curves = compact_conditions(curves)
        # Bound legend density, retaining every observed curve within the same PNG.
        for offset in range(0, len(curves), 8):
            part = panel | {"curves": curves[offset : offset + 8]}
            if len(curves) > 8:
                part["title"] += f" ({offset // 8 + 1})"
            panels[metadata[number - 1]["file"]].append(part)
    for entry in metadata:
        for key in ("series", "accepted_series", "sources"):
            entry[key] = sorted(set(entry[key]))
        entry["panels"] = len(panels[entry["file"]])
    return metadata, panels
