"""Streaming accepted-record SVG plotter; no producer logic or reaggregation."""

import argparse
import html
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from tqdm import tqdm

from mcpref.diagnostics.artifacts import (
    GROUP_PANELS,
    PLOT_REGISTRY,
    group_registry,
    iter_all_records,
    iter_records,
    pretrain_registry,
    stream_index,
)
from mcpref.shards.store import make_dirs, part_path, publish

COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")
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
    if key in record["metrics"]:
        return record["metrics"][key]
    joined = record["joined"]
    if key in joined.get("outputs", {}):
        return joined["outputs"][key]
    if key in joined.get("condition", {}).get("losses", {}):
        return joined["condition"]["losses"][key]
    if not key.startswith("preference."):
        return None
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


def svg(
    group: str, series: list[str], points: dict[str, list[tuple[float, float]]]
) -> str:
    width, height, margin = 960, 540, 60
    found = [point for values in points.values() for point in values]
    if found:
        xmin, xmax = min(x for x, _ in found), max(x for x, _ in found)
        ymin, ymax = min(y for _, y in found), max(y for _, y in found)
    else:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
    xspan, yspan = xmax - xmin or 1, ymax - ymin or 1

    def pixel(point):
        x, y = point
        return (
            margin + (x - xmin) / xspan * (width - 2 * margin),
            height - margin - (y - ymin) / yspan * (height - 2 * margin),
        )

    body = [
        f'<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="white" stroke="#94a3b8"/>',
        f'<text x="{margin}" y="30" font-family="sans-serif" font-size="20">{html.escape(group)}</text>',
        f'<text x="{margin}" y="{height - 15}" font-family="sans-serif" font-size="12">x={xmin:g}..{xmax:g}; y={ymin:g}..{ymax:g}</text>',
    ]
    for index, name in enumerate(series):
        values = points.get(name, [])
        if not values:
            continue
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in map(pixel, values))
        body.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        body.append(
            f'<text x="{width - 260}" y="{margin + 22 * (index + 1)}" fill="{color}" font-family="sans-serif" font-size="14">{html.escape(name)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(body)
        + "</svg>\n"
    )


def write_plot(path: Path, group: str, series: list[str], points: dict) -> None:
    make_dirs(path.parent)
    part = part_path(path)
    part.write_text(svg(group, series, points), encoding="utf-8")
    publish(part, path)


def selected(entry: dict, record: dict, name: str) -> bool:
    if entry["group"] == "target-provenance":
        model = name.startswith("target.model_")
        return record["kind"] == ("pretrain" if model else "targets")
    split, condition = entry["file"].split("__")[1:3]
    return (record["context"]["split"], record["context"]["condition"]) == (
        split,
        condition,
    )


def curve_points(record: dict, entry: dict, name: str) -> list[tuple[float, float]]:
    axis = entry["x"]
    if axis == "bin":
        return [
            (row["bin"], row[name])
            for row in record["joined"].get("reliability", [])
            if row[name] is not None
        ]
    if axis == "matched_mass":
        return [
            (row["mass"], float(row[name]))
            for row in record["joined"].get("matched_reliability", [])
        ]
    if axis.startswith("km:"):
        milestone = axis.removeprefix("km:")
        rows = record["joined"].get("kaplan_meier", {}).get(milestone, [])
        return [(row["time"], row[name]) for row in rows]
    point = xy(record, name, axis)
    return [] if point is None else [point]


def collect(records: Iterable[dict], entries: Sequence[dict]) -> dict:
    points = {
        entry["file"]: {name: [] for name in entry["series"]} for entry in entries
    }
    for record in records:
        for entry in entries:
            for name in entry["series"]:
                if not selected(entry, record, name):
                    continue
                points[entry["file"]][name].extend(curve_points(record, entry, name))
    for series in points.values():
        for values in series.values():
            values.sort()
    return points


def facets(
    records: Iterable[dict], series: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    found = set()
    for record in records:
        if series is None or any(source(record, name) is not None for name in series):
            found.add((record["context"]["split"], record["context"]["condition"]))
    return sorted(found)


def write_group(plot_dir: Path, entries: Sequence[dict], points: dict) -> None:
    for entry in entries:
        parts = entry["file"].split("__")
        group = (
            entry["group"]
            if len(parts) == 2
            else "/".join(part.removesuffix(".svg") for part in parts[1:])
        )
        write_plot(
            plot_dir / entry["file"], group, entry["series"], points[entry["file"]]
        )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument("plot_dir", type=Path)
    args = parser.parse_args(argv)
    groups = ("target-provenance", "pretrain-preference", *GROUP_PANELS)
    found = facets(iter_records(args.diagnostics_dir, "pretrain"))
    registries = {
        "target-provenance": list(PLOT_REGISTRY),
        "pretrain-preference": pretrain_registry(found),
    }
    generic_facets = {group: set() for group in GROUP_PANELS}
    update_groups = set()
    for record in iter_all_records(args.diagnostics_dir):
        facet = (record["context"]["split"], record["context"]["condition"])
        update_groups.update(
            name for name in record["metrics"] if name.startswith("update.group.")
        )
        for group, panels in GROUP_PANELS.items():
            series = [name for panel in panels.values() for name in panel]
            if any(source(record, name) is not None for name in series):
                generic_facets[group].add(facet)
    record = None
    for group in GROUP_PANELS:
        registries[group] = group_registry(
            group,
            sorted(generic_facets[group]),
            sorted(update_groups) if group == "update-health" else (),
        )
    plots = [entry for group in groups for entry in registries[group]]
    points = collect(iter_all_records(args.diagnostics_dir), plots)
    with tqdm(
        total=len(groups), desc="diagnostics plot", unit="group", file=sys.stderr
    ) as bar:
        for group in groups:
            write_group(args.plot_dir, registries[group], points)
            bar.update()
    stream_index(args.diagnostics_dir, iter_all_records(args.diagnostics_dir), plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
