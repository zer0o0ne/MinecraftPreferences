"""Interface Produces of specs/gui-masks.md: templates, report.json, report.md and
features.npz; `write_report` also used by specs/gui-stack.md."""

import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from mcpref.gui.label_files import OTHER
from mcpref.recognition.drawn_masks import SCALE
from mcpref.recognition.frame_set import frame_values
from mcpref.recognition.metrics import evaluation
from mcpref.recognition.pairs import Named
from mcpref.recognition.recognise import Point

PARTS = ("train", "validation", "test")


def template_image(v: str, t: np.ndarray) -> Image.Image:
    """Interface: RGB template; (f): 8-bit grey floor(T / 1000)."""
    if v == "f":
        return Image.fromarray((t[..., 0] // SCALE).astype(np.uint8))
    return Image.fromarray(t)


def write_templates(out: Path, named: Mapping[str, Named]) -> None:
    """Interface: templates/<variant>/<name>_template.png and _mask.png (255 in M)."""
    for v, classes in named.items():
        folder = out / "templates" / v
        folder.mkdir(parents=True, exist_ok=True)
        for name, (t, m) in ((n, p) for c in classes.values() for n, p in c.items()):
            template_image(v, t).save(folder / f"{name}_template.png", format="PNG")
            mask = Image.fromarray(m.astype(np.uint8) * 255)
            mask.save(folder / f"{name}_mask.png", format="PNG")


def split_report(
    parts: dict[str, int],
    part: list[list[dict]],
    labels: list[str],
    fold_of: dict[str, int],
    n_folds: int,
) -> dict:
    """Interface report.json `split`: episodes, frames F^part and frames per label; per
    fold, its training episodes and labelled frames F^tr_(l)."""
    report = {
        name: {
            "episodes": sum(p == i for p in parts.values()),
            "frames": len(part[i]),
            "per_label": {k: sum(f["label"] == k for f in part[i]) for k in labels},
        }
        for i, name in enumerate(PARTS)
    }
    in_fold = [fold_of[f["relpath"]] for f in part[0]]
    report["folds"] = [
        {"episodes": list(fold_of.values()).count(l), "frames": in_fold.count(l)}
        for l in range(n_folds)
    ]
    return report


def pairs_report(named: Named, names: list[str]) -> dict:
    """Interface report.json `pairs`: per class, the number of pairs and mask sizes."""
    sizes = {k: [int(m.sum()) for _, m in named.get(k, {}).values()] for k in names}
    return {k: {"n_pairs": len(s), "mask_sizes": s} for k, s in sizes.items()}


def variant_report(
    named: Named,
    chosen: dict[str, Point],
    preds: dict[str, list[str]],
    test: list[dict],
    names: list[str],
) -> dict:
    """Interface report.json, one variant: pairs; thresholds and (7) per choice."""
    ids, y, g = frame_values(test)
    return {"pairs": pairs_report(named, names)} | {
        c: {
            "thresholds": dict(zip(("tau", "theta", "mu"), point)),
            "test": evaluation(y, preds[c], ids, g, names + [OTHER]),
        }
        for c, point in chosen.items()
    }


def strings(values: Sequence[str]) -> np.ndarray:
    """Interface features.npz: a fixed-width unicode array."""
    return np.array(values, dtype=str)


def write_features(
    out: Path,
    named: Mapping[str, Named],
    labels: list[str],
    parts: Mapping[str, tuple[list[dict], np.ndarray]],
    preds: Mapping[str, Mapping[str, list[str]]],
) -> None:
    """Interface: features.npz (numpy.savez, no object arrays); columns
    <variant>/<class>[__<member>] in the order of `named` (14)."""
    columns = [f"{v}/{n}" for v, c in named.items() for ps in c.values() for n in ps]
    arrays = {"columns": strings(columns), "labels": strings(labels)}
    for part, (frames, x) in parts.items():
        ids, y, g = frame_values(frames)
        arrays[f"{part}_ids"], arrays[f"{part}_y"] = strings(ids), strings(y)
        arrays[f"{part}_g"] = np.array(g, dtype=bool)
        arrays[f"{part}_x"] = x
    for v, by_choice in preds.items():
        for c, y_hat in by_choice.items():
            arrays[f"test_pred_{v}_{c}"] = strings(y_hat)
    np.savez(out / "features.npz", **arrays)


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


def keyed(v: object) -> object:
    """Interface report.md: a non-empty list of dicts (`split` `folds`) as a dict keyed
    by position, so that it becomes a table."""
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        return {str(i): x for i, x in enumerate(v)}
    return v


def matrix_table(entry: dict) -> list[str]:
    """(8): A_k with the labellers as row and column headers; none when F_k is empty."""
    if entry["matrix"] is None:
        return []
    rows = [[a, *r] for a, r in zip(entry["labellers"], entry["matrix"])]
    return table(["", *entry["labellers"]], rows)


def markdown(value: dict, path: list[str], title: str) -> list[str]:
    """Interface report.md: report.json as sections of tables, headed `# <title>`."""
    value = {k: keyed(v) for k, v in value.items()}
    lines = ["## " + " / ".join(path) if path else f"# {title}", ""]
    if value and all(flat(v) for v in value.values()):
        cols = list(next(iter(value.values())))
        return lines + table(["", *cols], [[k, *v.values()] for k, v in value.items()])
    if "matrix" in value:
        return lines + [f"n = {value['n']}", ""] + matrix_table(value)
    scalars = [[k, v] for k, v in value.items() if not isinstance(v, dict)]
    lines += table(["key", "value"], scalars) if scalars else []
    for k, v in value.items():
        if isinstance(v, dict):
            lines += markdown(v, path + [k], title)
    return lines


def write_report(out: Path, report: dict, title: str) -> None:
    """Interface: {out}/report.json and report.md headed `# <title>`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (out / "report.md").write_text("\n".join(markdown(report, [], title)))
