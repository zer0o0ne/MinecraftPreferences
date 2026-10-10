"""Interface Produces of specs/gui-masks.md: templates, report.json and report.md."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from mcpref.gui.masks import Pair

PARTS = ("train", "validation", "test")
Named = dict[str, dict[str, Pair]]


def write_templates(out: Path, named: dict[str, Named]) -> None:
    """Interface: templates/<variant>/<name>_template.png (RGB) and _mask.png (255 in M)."""
    for v, classes in named.items():
        folder = out / "templates" / v
        folder.mkdir(parents=True, exist_ok=True)
        for name, (t, m) in ((n, p) for c in classes.values() for n, p in c.items()):
            Image.fromarray(t).save(folder / f"{name}_template.png", format="PNG")
            mask = Image.fromarray(m.astype(np.uint8) * 255)
            mask.save(folder / f"{name}_mask.png", format="PNG")


def split_report(
    parts: dict[str, int], part: list[list[dict]], labels: list[str]
) -> dict:
    """Interface report.json `split`: episodes, frames F^part and frames per label."""
    return {
        name: {
            "episodes": sum(p == i for p in parts.values()),
            "frames": len(part[i]),
            "per_label": {k: sum(f["label"] == k for f in part[i]) for k in labels},
        }
        for i, name in enumerate(PARTS)
    }


def pairs_report(named: Named, names: list[str]) -> dict:
    """Interface report.json `pairs`: per class, the number of pairs and mask sizes."""
    sizes = {k: [int(m.sum()) for _, m in named.get(k, {}).values()] for k in names}
    return {k: {"n_pairs": len(s), "mask_sizes": s} for k, s in sizes.items()}


def cell(v: object) -> str:
    """Interface report.md: one table cell; ratios to four decimals."""
    if v is None:
        return "null"
    if isinstance(v, Fraction):
        return f"{float(v):.4f}"
    if isinstance(v, list):
        return ", ".join(map(cell, v))
    return str(v)


def table(header: list[str], rows: list[list]) -> list[str]:
    """Interface report.md: a Markdown table."""

    def line(cells: list) -> str:
        return "| " + " | ".join(cells) + " |"

    body = [line([cell(c) for c in row]) for row in rows]
    return [line(header), line(["---"] * len(header)), *body, ""]


def flat(v: object) -> bool:
    """A dict without nested dicts or a matrix: one table row."""
    return (
        isinstance(v, dict)
        and "matrix" not in v
        and not any(isinstance(x, dict) for x in v.values())
    )


def matrix_table(entry: dict) -> list[str]:
    """(8): A_k with the labellers as row and column headers; none when F_k is empty."""
    if entry["matrix"] is None:
        return []
    rows = [[a, *r] for a, r in zip(entry["labellers"], entry["matrix"])]
    return table(["", *entry["labellers"]], rows)


def markdown(value: dict, path: list[str]) -> list[str]:
    """Interface report.md: report.json as sections of tables."""
    lines = ["## " + " / ".join(path) if path else "# GUI masks report", ""]
    if value and all(flat(v) for v in value.values()):
        cols = list(next(iter(value.values())))
        return lines + table(["", *cols], [[k, *v.values()] for k, v in value.items()])
    if "matrix" in value:
        return lines + [f"n = {value['n']}", ""] + matrix_table(value)
    scalars = [[k, v] for k, v in value.items() if not isinstance(v, dict)]
    lines += table(["key", "value"], scalars) if scalars else []
    for k, v in value.items():
        if isinstance(v, dict):
            lines += markdown(v, path + [k])
    return lines


def write_report(out: Path, report: dict) -> None:
    """Interface: {out_dir}/report.json and report.md."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (out / "report.md").write_text("\n".join(markdown(report, [])))
