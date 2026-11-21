"""Deterministic two-pass selection of specs/pretrain-diagnostics.md (17)."""

import json
import sys

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.diagnostics.pretrain.diagnostics import GlobalFit
from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.training.pretrain.classical.config import PretrainConfig
from mcpref.training.pretrain.references.selection import (
    PARTS,
    selected_ordinals,
)


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
