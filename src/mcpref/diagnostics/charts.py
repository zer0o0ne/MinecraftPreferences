"""Headless Matplotlib figures from already computed scalar observations."""

import math
import textwrap
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from mcpref.infrastructure.shards.store import make_dirs, part_path, publish


def figure(title: str, count: int) -> Figure:
    columns = min(2, max(1, count))
    rows = math.ceil(max(1, count) / columns)
    result = Figure(figsize=(6 * columns, 3.4 * rows + 0.7), layout="constrained")
    FigureCanvasAgg(result)
    title = "\n".join(
        textwrap.fill(line, width=52 * columns) for line in title.splitlines()
    )
    result.suptitle(title, fontsize=13)
    for i in range(max(1, count)):
        axis = result.add_subplot(rows, columns, i + 1)
        axis.set_axisbelow(True)
        axis.grid(True, color="#dfe5ec", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    return result


def line_figure(title: str, points: dict, x_label: str, *, step: bool) -> Figure:
    result = figure(title, len(points))
    for axis, (metric, values) in zip(result.axes, points.items()):
        axis.set_title(textwrap.fill(metric, width=52), fontsize=10)
        axis.set_xlabel(x_label, fontsize=9)
        axis.set_ylabel("Recorded value", fontsize=9)
        if not any(math.isfinite(y) for _, y in values):
            axis.text(
                0.5, 0.5, "No observations", ha="center", transform=axis.transAxes
            )
            continue
        x, y = zip(*values)
        axis.plot(
            x,
            y,
            color="#2463a7",
            linewidth=1.4,
            marker=".",
            markersize=4,
            drawstyle="steps-post" if step else "default",
        )
        axis.margins(x=0.03, y=0.12)
        if metric in ("confidence", "accuracy", "survival", "hit"):
            axis.set_ylim(0, 1)
    return result


def save_png(result: Figure, path: Path) -> None:
    make_dirs(path.parent)
    part = part_path(path)
    try:
        result.savefig(part, format="png", dpi=120, facecolor="white")
        publish(part, path)
    finally:
        result.clear()
