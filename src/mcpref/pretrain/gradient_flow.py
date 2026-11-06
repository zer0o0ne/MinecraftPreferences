"""Selected-step replay and accepted gradient-flow records of (22)-(29)."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref import REPO
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.gradient_pairs import (
    baseline_pairs,
    counterfactual_metrics,
    labelled_items,
    pair_distribution,
    pair_records,
    post_step_pairs,
)
from mcpref.pretrain.gradient_summary import (
    METRIC_PATHS,
    step_distribution,
    summarize,
)
from mcpref.pretrain.items import Part, read_items
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards, shard_path
from mcpref.training.schedule import remaining

__all__ = [
    "METRIC_PATHS",
    "baseline_pairs",
    "counterfactual_metrics",
    "labelled_items",
    "pair_distribution",
    "pair_records",
    "post_step_pairs",
    "step_distribution",
    "summarize",
]

START = {"step": 0, "epoch": 1, "schedule_rng": None, "batch_in_epoch": 0}
ROLES = ("beginning", "middle", "end")


@dataclass(frozen=True)
class GradientStepRecord:
    metrics: dict
    pairs: list[dict]


def gradient_dir(cfg: PretrainConfig) -> Path:
    return REPO / cfg.out_dir / "gradient-flow"


def select_gradient_steps(cfg: PretrainConfig, run: dict, data: Part) -> dict:
    """(22): replay the real schedule with a fresh local PCG64 generator."""
    path = gradient_dir(cfg) / "selection.json"
    if not run["counts"]["train"]["pairs"]:
        print("pretrain gradient flow: nothing to do", file=sys.stderr)
        return {"version": 1, "middle_target": max(1, data.total // 2), "steps": []}
    if path.exists():
        print("pretrain gradient flow selection: already done", file=sys.stderr)
        return json.loads(path.read_bytes())
    generator = np.random.Generator(np.random.PCG64([cfg.seed, 3]))
    labelled, visit, items = [], None, None
    with tqdm(
        total=data.total,
        desc="pretrain gradient flow selection",
        unit="step",
        file=sys.stderr,
        mininterval=0,
    ) as bar:
        bar.refresh()
        for _, shard, positions in remaining(
            generator, data.sizes, cfg.batch, START, cfg.epochs
        ):
            if visit != shard:
                visit, items = shard, read_items(cfg, shard)
            if any("labels" in items[index] for index in positions):
                labelled.append(bar.n + 1)
            bar.update()
            bar.refresh()
    middle_target = max(1, data.total // 2)
    beginning = labelled[0]
    end = labelled[-1]
    middle = next((step for step in labelled if step >= middle_target), None)
    if middle is None:
        middle = max(step for step in labelled if step < middle_target)
    roles = {}
    for role, step in zip(ROLES, (beginning, middle, end)):
        roles.setdefault(step, []).append(role)
    selection = {
        "version": 1,
        "middle_target": middle_target,
        "steps": [
            {"step": step, "roles": [role for role in ROLES if role in names]}
            for step, names in sorted(roles.items())
        ],
    }
    write_json(path, selection)
    return selection


def merge_alignment(reduced: dict, alignment: dict) -> dict:
    unavailable_groups = {
        group
        for bundles in alignment.values()
        for group, value in bundles.items()
        if value["update_unavailable"] is not None
    }
    for x in ("phi", "P"):
        for group, bundle in reduced[x]["groups"].items():
            value = alignment[x][group]
            bundle["update_dot"] = (
                None if group in unavailable_groups else value["update_dot"]
            )
            bundle["update_unavailable"] = (
                "zero_decay_factor"
                if group in unavailable_groups
                else value["update_unavailable"]
            )
        values = [bundle for bundle in reduced[x]["groups"].values()]
        unavailable = next(
            (
                bundle["update_unavailable"]
                for bundle in values
                if bundle["update_unavailable"]
            ),
            None,
        )
        dots = [bundle["update_dot"] for bundle in values]
        reduced[x]["total"]["update_dot"] = (
            sum(dots) if all(dot is not None for dot in dots) else None
        )
        reduced[x]["total"]["update_unavailable"] = unavailable
    return reduced


def accepted_records(cfg: PretrainConfig) -> list[GradientStepRecord]:
    directory = REPO / cfg.out_dir / "checkpoints"
    records = []
    for checkpoint in range(complete_shards(directory)):
        root = shard_path(directory, checkpoint) / "gradient_flow"
        if not root.exists():
            continue
        for shard in sorted(root.iterdir()):
            records.append(
                GradientStepRecord(
                    json.loads((shard / "metrics.json").read_bytes()),
                    json.loads((shard / "pairs.json").read_bytes()),
                )
            )
    return records
