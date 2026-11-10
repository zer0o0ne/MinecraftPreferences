"""Deterministic two-pass selection of specs/pretrain-diagnostics.md (17)."""

import json
import math
import sys

import numpy as np
from tqdm import tqdm

from mcpref.config import REPO
from mcpref.diagnostics.pretrain.diagnostics import GlobalFit
from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.training.pretrain.config import PretrainConfig

PARTS = ("train", "validation")


def part_pairs(fit: GlobalFit, part: str) -> int:
    return sum(getattr(stratum, part) for stratum in fit.strata)


def allocations(fit: GlobalFit, part: str, pairs: int) -> dict[str, int]:
    """(17): largest-remainder allocation in catalogue order."""
    total = part_pairs(fit, part)
    if total < pairs:
        raise RuntimeError(f"{part} has only {total} pairs, reference needs {pairs}")
    exact = [pairs * getattr(stratum, part) / total for stratum in fit.strata]
    result = {
        stratum.key: math.floor(value) for stratum, value in zip(fit.strata, exact)
    }
    remaining = pairs - sum(result.values())
    order = sorted(
        range(len(fit.strata)),
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in order[:remaining]:
        result[fit.strata[index].key] += 1
    return result


def selected_ordinals(cfg: PretrainConfig, fit: GlobalFit) -> dict:
    """One local PCG64 stream in part/stratum order, with sorted source ordinals."""
    generator = np.random.Generator(np.random.PCG64([cfg.seed, 4]))
    selected = {}
    for part in PARTS:
        allocation = allocations(fit, part, cfg.reference_pairs)
        selected[part] = {}
        for stratum in fit.strata:
            count = getattr(stratum, part)
            chosen = generator.permutation(count)[: allocation[stratum.key]]
            selected[part][stratum.key] = sorted(chosen.tolist())
    return selected


def scan_shard(
    items: list[dict],
    part: str,
    shard: int,
    offsets: dict,
    chosen: dict,
    found: dict,
) -> None:
    """Retain selected stratum ordinals from one index shard and advance offsets."""
    local = dict.fromkeys(offsets[part], 0)
    for item_index, item in enumerate(items):
        if "labels" not in item:
            continue
        labels = item["labels"]
        key = "inc" if labels["inc"] else json.dumps(labels["p"])
        ordinal = offsets[part][key] + local[key]
        local[key] += 1
        if ordinal in chosen[key]:
            found[part][key].append(
                (
                    ordinal,
                    item
                    | {
                        "part": part,
                        "stratum": key,
                        "shard": shard,
                        "item_index": item_index,
                    },
                )
            )
    for key, count in local.items():
        offsets[part][key] += count


def select_items(cfg: PretrainConfig, fit: GlobalFit) -> dict:
    """(17): pass 2, reading one index shard at a time in part order."""
    selected = selected_ordinals(cfg, fit)
    found = {part: {stratum.key: [] for stratum in fit.strata} for part in PARTS}
    offsets = {part: {stratum.key: 0 for stratum in fit.strata} for part in PARTS}
    directories = {part: REPO / cfg.index_dir / part for part in PARTS}
    shards = {part: complete_shards(directories[part]) for part in PARTS}
    with tqdm(
        total=sum(shards.values()),
        desc="pretrain reference select",
        unit="shard",
        file=sys.stderr,
    ) as bar:
        for part in PARTS:
            chosen = {key: set(values) for key, values in selected[part].items()}
            for shard in range(shards[part]):
                path = shard_path(directories[part], shard) / "items.json"
                items = json.loads(path.read_bytes())["items"]
                scan_shard(items, part, shard, offsets, chosen, found)
                bar.update()
    items = [
        record
        for part in PARTS
        for stratum in fit.strata
        for _, record in sorted(found[part][stratum.key])
    ]
    return {
        "version": 1,
        "seed": cfg.seed,
        "pairs_per_part": cfg.reference_pairs,
        "items": items,
    }
