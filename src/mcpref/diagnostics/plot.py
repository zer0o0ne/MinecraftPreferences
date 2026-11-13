"""Read saved diagnostics and render a local Matplotlib report."""

import argparse
import hashlib
import math
import sys
from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path

from tqdm import tqdm

from mcpref.diagnostics.charts import save_png
from mcpref.diagnostics.core.artifacts import (
    iter_all_records,
    stream_index,
)
from mcpref.diagnostics.gallery import write_gallery
from mcpref.diagnostics.report.build import dashboards
from mcpref.diagnostics.report.gradient import gradient_records
from mcpref.diagnostics.report.render import dashboard_figure
from mcpref.diagnostics.saved import iter_saved_runs
from mcpref.diagnostics.series import collect, registries


def saved_points(records: Iterable[dict]) -> tuple[list[dict], dict]:
    groups = {}
    source_steps = {}
    for record in records:
        source_steps.setdefault((record["group"], record["source"]), set()).add(
            record["step"]
        )
        for metric, value in record["values"].items():
            parts = metric.split(".")
            if len(parts) == 1:
                panel = "losses"
            elif parts[0] in (
                "components",
                "outputs",
                "gradients",
                "updates",
                "preference",
                "coverage",
            ):
                panel = ".".join(parts[:2])
            else:
                panel = parts[0]
            key = (record["group"], record["source"], panel)
            title, points = groups.setdefault(key, (record["title"], {}))
            points.setdefault(metric, []).append(
                (record["step"], math.nan if value is None else value)
            )
    entries, all_points = [], {}
    for (group, source_path, panel), (title, points) in groups.items():
        metrics = sorted(points)
        for offset in range(0, len(metrics), 6):
            names = metrics[offset : offset + 6]
            suffix = hashlib.sha256(f"{source_path}\0{panel}".encode()).hexdigest()[:12]
            filename = f"{group}__{suffix}__{offset // 6}.png"
            entries.append(
                {
                    "group": group,
                    "file": filename,
                    "series": names,
                    "x": "step",
                    "title": f"{title} | {panel}\n{source_path}",
                    "source": source_path,
                    "population": title,
                    "x_label": "Selected optimizer step"
                    if group == "saved-gradient-flow"
                    else "Optimizer step",
                }
            )
            steps = sorted(source_steps[group, source_path])
            all_points[filename] = {}
            for name in names:
                lookup = dict(points[name])
                all_points[filename][name] = [
                    (step, lookup.get(step, math.nan)) for step in steps
                ]
    return entries, all_points


def render(directory: Path, entries: list[dict], points: dict) -> None:
    if not entries:
        print("diagnostics plot: nothing to do", file=sys.stderr)
        return
    for entry in tqdm(entries, desc="diagnostics plot", unit="plot", file=sys.stderr):
        result = dashboard_figure(entry, points[entry["file"]])
        save_png(result, directory / entry["file"])


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics_dir", type=Path)
    parser.add_argument("plot_dir", type=Path)
    parser.add_argument("--pretrain-dir", type=Path)
    parser.add_argument("--rl-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.diagnostics_dir.is_dir() and not (args.pretrain_dir or args.rl_dir):
        parser.error(f"diagnostics directory does not exist: {args.diagnostics_dir}")
    entries = registries(iter_all_records(args.diagnostics_dir))
    points = collect(iter_all_records(args.diagnostics_dir), entries)
    extra, saved = saved_points(
        chain(
            iter_saved_runs(args.pretrain_dir, args.rl_dir),
            gradient_records(args.pretrain_dir),
        )
    )
    report, panels = dashboards(entries + extra, points | saved)
    render(args.plot_dir, report, panels)
    if args.diagnostics_dir.is_dir():
        registry = [
            {key: entry[key] for key in ("group", "file", "series", "x")}
            | {"series": entry["accepted_series"]}
            for entry in report
            if entry["accepted_series"]
        ]
        stream_index(
            args.diagnostics_dir, iter_all_records(args.diagnostics_dir), registry
        )
    write_gallery(args.plot_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
