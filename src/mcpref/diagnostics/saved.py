"""Stream persisted training scalars without loading models or recomputing metrics."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

from mcpref.infrastructure.shards.store import (
    complete_shards,
    index,
    is_directory,
    shard_path,
)

STEP_KEYS = (
    "phi",
    "pi",
    "P",
    "loss",
    "ppo",
    "shift",
    "metra",
    "lr",
    "counts",
    "components",
    "outputs",
    "phi_stats",
    "gradients",
    "updates",
    "baselines",
)
WINDOW_KEYS = (
    "losses",
    "coverage",
    "counts",
    "preference",
    "pi",
    "phi",
    "gradients",
    "updates",
)
REFERENCE_KEYS = ("losses", "preference", "baselines", "pi", "phi")


def scalars(value: dict, prefix: str = "") -> dict[str, float | None]:
    """Retain scalar dictionary leaves verbatim; vectors and metadata are not curves."""
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(scalars(item, path))
        elif item is None or type(item) in (int, float):
            result[path] = item
    return result


def record(
    group: str, title: str, source: Path, step: int, values: dict, source_file: Path
) -> dict:
    return {
        "group": group,
        "title": title,
        "source": str(source),
        "source_file": str(source_file),
        "x_label": "Optimizer step",
        "step": step,
        "values": scalars(values),
    }


def checkpoints(root: Path, kind: str, title: str) -> Iterator[dict]:
    directory = root / "checkpoints"
    count = complete_shards(directory)
    if not count:
        return
    for k in tqdm(range(count), desc=title, unit="checkpoint", file=sys.stderr):
        shard = shard_path(directory, k)
        files = [("losses.json", "steps", "step", STEP_KEYS)]
        if kind == "pretrain" and (shard / "diagnostics.json").is_file():
            files.append(("diagnostics.json", "windows", "last_step", WINDOW_KEYS))
        for filename, event, step_key, keys in files:
            for row in json.loads((shard / filename).read_bytes()):
                values = {key: row[key] for key in keys if key in row}
                yield record(
                    f"saved-{kind}-{event}",
                    f"{title} / {event}",
                    root,
                    row[step_key],
                    values,
                    shard / filename,
                )


def references(root: Path, title: str) -> Iterator[dict]:
    directory = root / "reference"
    count = complete_shards(directory)
    if not count:
        return
    for k in tqdm(
        range(count), desc=f"{title} reference", unit="shard", file=sys.stderr
    ):
        path = shard_path(directory, k) / "metrics.json"
        row = json.loads(path.read_bytes())
        for condition, parts in row["conditions"].items():
            for split, metrics in parts.items():
                values = {key: metrics[key] for key in REFERENCE_KEYS if key in metrics}
                yield record(
                    "saved-pretrain-reference",
                    f"{title} / {split} / {condition}",
                    directory / split / condition,
                    row["step"],
                    values,
                    path,
                )


def run_roots(
    pretrain_dir: Path | None, rl_dir: Path | None
) -> Iterator[tuple[Path, str, str]]:
    if pretrain_dir is not None:
        yield pretrain_dir, "pretrain", "Pretrain / joint"
        if (pretrain_dir / "retrain").is_dir():
            yield pretrain_dir / "retrain", "pretrain", "Pretrain / retrain"
    if rl_dir is not None:
        cycles = sorted(
            (
                path
                for path in rl_dir.iterdir()
                if index(path.name) is not None and is_directory(path)
            ),
            key=lambda path: index(path.name),
        )
        for cycle in cycles:
            title = f"RL cycle {index(cycle.name)}"
            yield cycle, "rl", f"{title} / joint"
            if (cycle / "retrain").is_dir():
                yield cycle / "retrain", "rl", f"{title} / retrain"


def iter_saved_runs(pretrain_dir: Path | None, rl_dir: Path | None) -> Iterator[dict]:
    """Yield separate step/window/reference streams from complete source shards."""
    if pretrain_dir is None and rl_dir is None:
        return
    for root in (pretrain_dir, rl_dir):
        if root is not None and not root.is_dir():
            if not root.exists():
                raise FileNotFoundError(root)
            raise NotADirectoryError(root)
    found = False
    for root, kind, title in run_roots(pretrain_dir, rl_dir):
        for row in checkpoints(root, kind, title):
            found = True
            yield row
        if kind == "pretrain":
            for row in references(root, title):
                found = True
                yield row
    if not found:
        print("saved training logs: no complete records", file=sys.stderr)
