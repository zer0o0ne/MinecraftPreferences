"""Readable shared-scale panels and explicit no-data cards, rendered headlessly."""

import math
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from mcpref.diagnostics.report.routing import canonical

COLORS = (
    "#2563a6",
    "#d26922",
    "#25866d",
    "#9c4697",
    "#8b7126",
    "#5786a3",
    "#c14450",
    "#687344",
)


def draw_panel(axis, panel: dict) -> None:
    title = panel["title"] + "\n" + panel["population"]
    if panel["snapshot"]:
        title += f"\nLatest snapshot: {panel['snapshot']}"
    axis.set_title(title, fontsize=11, loc="left", pad=12)
    axis.set_xlabel(panel["x_label"], fontsize=10)
    axis.set_ylabel(panel["unit"], fontsize=10)
    axis.grid(True, color="#dde3eb", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    colors = {}
    windowed = any(curve["mode"] == "Windows" for curve in panel["curves"])
    for curve in panel["curves"]:
        color_key = (canonical(curve["metric"]), curve["condition"], curve["cycle"])
        color = colors.setdefault(color_key, COLORS[len(colors) % len(COLORS)])
        x, y = zip(*curve["points"])
        label = textwrap.fill(curve["label"], width=42)
        axis.plot(
            x,
            y,
            label=label,
            color=color,
            linewidth=2 if curve["mode"] == "Windows" else 1.4,
            alpha=0.5 if windowed and curve["mode"] == "Steps" else 1,
            marker="x" if curve["split"] == "validation" else ".",
            markersize=4,
            linestyle="--"
            if curve["split"] == "validation" or curve["mode"] == "Windows"
            else "-",
            drawstyle="steps-post" if panel["step"] else "default",
        )
    axis.margins(x=0.03, y=0.12)
    if panel["unit"].startswith("Fraction") or panel["unit"] in (
        "Probability",
        "Probability mass",
        "Probability / error",
        "Probability error",
        "Not yet reached (fraction)",
        "Active flag",
    ):
        axis.set_ylim(0, 1)
    elif panel["unit"] == "Cosine":
        axis.set_ylim(-1, 1)
    x_values = {x for curve in panel["curves"] for x, _ in curve["points"]}
    if len(x_values) == 1:
        axis.set_xticks(sorted(x_values))
    elif panel["x_label"] != "Matched probability mass":
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.tick_params(labelsize=9)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.24),
        borderaxespad=0,
        frameon=False,
        fontsize=9,
        ncol=2 if len(panel["curves"]) > 3 else 1,
    )


def dashboard_figure(entry: dict, panels: list[dict]) -> Figure:
    columns = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    height = 4 if not panels else 5.6 * rows + 1.2
    result = Figure(figsize=(6.6 * columns, height), layout="constrained")
    FigureCanvasAgg(result)
    subtitle = entry["description"]
    if entry["missing"]:
        subtitle += f"\n{entry['missing']} unavailable series omitted; missing values are not zeros."
    result.suptitle(f"{entry['title']}\n{subtitle}", fontsize=14, ha="left", x=0.03)
    if not panels:
        result.text(0.06, 0.55, "No recorded observations", fontsize=16, weight="bold")
        result.text(
            0.06, 0.24, textwrap.fill(entry["empty_reason"], width=67), fontsize=11
        )
        return result
    for i, panel in enumerate(panels):
        draw_panel(result.add_subplot(rows, columns, i + 1), panel)
    return result
