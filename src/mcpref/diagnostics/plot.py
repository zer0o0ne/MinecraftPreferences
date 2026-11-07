"""Streaming accepted-record SVG plotter; no producer logic or reaggregation."""

import argparse
import html
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from tqdm import tqdm

from mcpref.diagnostics.artifacts import (
    PLOT_REGISTRY,
    iter_all_records,
    iter_records,
    pretrain_registry,
    stream_index,
)
from mcpref.shards.store import make_dirs, part_path, publish

COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")


def at(value: dict, path: str):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def source(record: dict, key: str):
    if key.startswith("joined."):
        return at(record, key)
    if not key.startswith("preference."):
        return record["metrics"].get(key)
    path = key.removeprefix("preference.")
    joined = record["joined"]
    if "condition" in joined:
        return at(joined["condition"]["preference"], path)
    window = joined.get("window_preference")
    return at(window, path) if window is not None else None


def scalar(value):
    if isinstance(value, dict):
        return value.get("value", value.get("mean"))
    return value if isinstance(value, (int, float)) else None


def xy(record: dict, key: str, axis: str):
    value = scalar(source(record, key))
    if value is None:
        return None
    if axis == "index":
        x = record["index"]
    else:
        steps = record["context"]["steps"]
        if len(steps) != 1:
            raise RuntimeError(
                "pretrain-preference requires exactly one optimizer step"
            )
        x = steps[0]
    return x, value


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
    xspan = xmax - xmin or 1
    yspan = ymax - ymin or 1

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


def collect(records: Iterable[dict], entries: Sequence[dict]) -> dict:
    points = {
        entry["file"]: {name: [] for name in entry["series"]} for entry in entries
    }
    for record in records:
        for entry in entries:
            for name in entry["series"]:
                if not selected(entry, record, name):
                    continue
                point = xy(record, name, entry["x"])
                if point is not None:
                    points[entry["file"]][name].append(point)
    for series in points.values():
        for values in series.values():
            values.sort()
    return points


def facets(records: Iterable[dict]) -> list[tuple[str, str]]:
    return sorted(
        {
            (record["context"]["split"], record["context"]["condition"])
            for record in records
        }
    )


def write_group(plot_dir: Path, entries: Sequence[dict], points: dict) -> None:
    for entry in entries:
        split = entry["file"].split("__")
        group = (
            entry["group"]
            if len(split) == 2
            else "/".join(part.removesuffix(".svg") for part in split[1:])
        )
        write_plot(
            plot_dir / entry["file"], group, entry["series"], points[entry["file"]]
        )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument("plot_dir", type=Path)
    args = parser.parse_args(argv)
    found = facets(iter_records(args.diagnostics_dir, "pretrain"))
    preference_entries = pretrain_registry(found)
    with tqdm(total=2, desc="diagnostics plot", unit="group", file=sys.stderr) as bar:
        target_points = collect(iter_all_records(args.diagnostics_dir), PLOT_REGISTRY)
        write_group(args.plot_dir, PLOT_REGISTRY, target_points)
        bar.update()
        preference_points = collect(
            iter_records(args.diagnostics_dir, "pretrain"), preference_entries
        )
        write_group(args.plot_dir, preference_entries, preference_points)
        bar.update()
    plots = [*PLOT_REGISTRY, *preference_entries]
    stream_index(args.diagnostics_dir, iter_all_records(args.diagnostics_dir), plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
