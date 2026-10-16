"""(E) of specs/gui-masks.md: report.json and report.md from the state, the model and the
test shards (bar `report`); `variant_preds`, `write_report` also used by
specs/gui-stack.md."""

import json
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mcpref import REPO
from mcpref.gui.label_files import OTHER
from mcpref.recognition.frame_set import State
from mcpref.recognition.metrics import (
    CRITERIA,
    evaluation,
    matrices,
    tally,
    tally_matrices,
)
from mcpref.recognition.pairs import VARIANTS
from mcpref.recognition.stored import (
    ALL,
    PARTS,
    TAU_THETA_MU,
    Model,
    read_part,
)

if TYPE_CHECKING:
    from mcpref.recognition.evaluate import GuiMasksConfig


def split_report(state: State, labels: list[str], k_folds: int) -> dict:
    """Interface report.json `split`: per part, its episodes, labelled frames and frames
    per label; per fold, its training episodes and labelled frames F^tr_(l)."""
    report = {
        name: {
            "episodes": sum(p == i for p in state.parts.values()),
            "frames": sum(c[i] for c in state.counts),
            "per_label": {k: state.per_label[i][k] for k in labels},
        }
        for i, name in enumerate(PARTS)
    }
    in_fold = list(state.fold_of.values())
    report["folds"] = [
        {"episodes": in_fold.count(l), "frames": state.fold_frames[l]}
        for l in range(k_folds)
    ]
    return report


def pairs_report(own: dict, names: list[str]) -> dict:
    """Interface report.json `pairs`: per class, the number of pairs and mask sizes."""
    sizes = {k: [int(m.sum()) for _, m in own.get(k, {}).values()] for k in names}
    return {k: {"n_pairs": len(s), "mask_sizes": s} for k, s in sizes.items()}


def variant_preds(pred: np.ndarray) -> dict[str, dict[str, list[str]]]:
    """Interface features `pred`: variant -> threshold choice -> y_hat, in the orders
    VARIANTS, CRITERIA; also used by specs/gui-stack.md (6)."""
    return {
        v: {c: pred[:, i, j].tolist() for j, c in enumerate(CRITERIA)}
        for i, v in enumerate(VARIANTS)
    }


def tally_shard(acc: dict, matrix_acc: dict, z: dict) -> None:
    """(7), (8): adds one test shard, its predictions of each variant and choice."""
    y, ids, g = z["y"].tolist(), z["ids"].tolist(), z["g"].tolist()
    preds = variant_preds(z["pred"])
    for v in VARIANTS:
        for c in CRITERIA:
            tally(acc[v][c], y, preds[v][c], ids, g)
    tally_matrices(matrix_acc, y, preds)


def accumulated(cfg: "GuiMasksConfig", state: State) -> tuple[dict, dict]:
    """(E): (7), (8) accumulated over the test shards read one at a time; bar `report`."""
    acc, matrix_acc = {v: {c: {} for c in CRITERIA} for v in VARIANTS}, {}
    tally = partial(tally_shard, acc, matrix_acc)
    read_part(REPO / cfg.out_dir, 2, state.counts, "report", tally)
    return acc, matrix_acc


def report(cfg: "GuiMasksConfig", state: State, model: Model, chosen: dict) -> dict:
    """(E): report.json; the split from the state, `pairs` from the model."""
    acc, matrix_acc = accumulated(cfg, state)
    labels = [*state.names, OTHER]
    out = {"split": split_report(state, labels, cfg.folds), "qualified": state.q}
    for v in VARIANTS:
        out[v] = {"pairs": pairs_report(model[ALL][v], state.names)} | {
            c: {
                "thresholds": dict(zip(TAU_THETA_MU, chosen[v][c])),
                "test": evaluation(acc[v][c], labels),
            }
            for c in CRITERIA
        }
    return out | {"agreement_matrices": matrices(matrix_acc, state.names)}


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
