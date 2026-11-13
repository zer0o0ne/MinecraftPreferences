"""Read accepted selected-step metrics of specs/pretrain-diagnostics.md (28)."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from mcpref.diagnostics.saved import record, run_roots
from mcpref.infrastructure.shards.store import (
    complete_shards,
    index,
    is_directory,
    shard_path,
)


def selected_record(
    root: Path, title: str, step: int, values: dict, path: Path
) -> dict:
    row = record(
        "saved-gradient-flow", f"{title} / selected steps", root, step, values, path
    )
    row["x_label"] = "Selected optimizer step"
    row["values"] = {
        key: value
        for key, value in row["values"].items()
        if not key.endswith(".update_unavailable")
    }
    return row


def summary_records(root: Path, title: str) -> Iterator[dict]:
    path = root / "gradient-flow/summary.json"
    if not path.is_file():
        return
    for row in json.loads(path.read_bytes())["steps"]:
        values = {
            "phi": {"total": row["phi_total"]},
            "P": {"total": row["P_total"]},
            "counterfactual": {
                key: {"mean": row[f"{key}_mean"]}
                for key in ("delta_only_phi", "delta_joint")
            },
        }
        yield selected_record(root, title, row["step"], values, path)


def accepted_records(root: Path, title: str) -> Iterator[dict]:
    checkpoints = root / "checkpoints"
    count = complete_shards(checkpoints)
    if not count:
        return
    for k in tqdm(
        range(count), desc=f"{title} selected steps", unit="checkpoint", file=sys.stderr
    ):
        directory = shard_path(checkpoints, k) / "gradient_flow"
        if not directory.is_dir():
            continue
        for step_dir in sorted(directory.iterdir()):
            if index(step_dir.name) is None or not is_directory(step_dir):
                continue
            path = step_dir / "metrics.json"
            row = json.loads(path.read_bytes())
            if row.get("evidence", {}).get("kind") != "real_pretrain_step":
                continue
            values = {
                key: row[key] for key in ("phi", "P", "nonfinite", "counterfactual")
            }
            yield selected_record(root, title, row["step"], values, path)


def gradient_records(pretrain_dir: Path | None) -> Iterator[dict]:
    """Prefer detailed accepted records; summaries retain totals when shards are absent."""
    for root, _, title in run_roots(pretrain_dir, None):
        rows = {row["step"]: row for row in summary_records(root, title)}
        rows.update((row["step"], row) for row in accepted_records(root, title))
        for step in sorted(rows):
            yield rows[step]
