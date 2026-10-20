"""(I), (1)-(3), (4) of specs/train-index.md: the inputs read in one pass with their
fingerprints, the items of U and R, the split by groups of episodes, the shuffle."""

import hashlib
import json
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

from mcpref import REPO
from mcpref.recognition.stored import shard_bar
from mcpref.shards.store import complete_shards, shard_path

if TYPE_CHECKING:
    from mcpref.pretrain.index import TrainIndexConfig

PARTS = ("train", "validation")
RECORDS, PAIRS = "records.json", "pairs.json"
Entry = tuple[dict, tuple[str, ...]]
"""An item and the episodes of its clips: its clip's, or a's then b's."""


@cache
def clips_of(directory: str, k: int) -> PurePosixPath:
    """Notation p without its last `/ id`: <directory>/shards/<name(k)>/clips."""
    return shard_path(PurePosixPath(directory) / "shards", k) / "clips"


def clip_path(directory: str, k: int, clip_id: str) -> str:
    """Notation p: <directory>/shards/<name(k)>/clips/<id>, relative to the root."""
    return str(clips_of(directory, k) / clip_id)


def unlabeled(directory: str, k: int, records: list[dict]) -> list[Entry]:
    """Notation U: the clips of shard k's records in order, null `clips` skipped."""
    return [
        ({"clip": clip_path(directory, k, i)}, (r["relpath"],))
        for r in records
        if r["clips"] is not None
        for i in r["clips"]
    ]


def labeled(directory: str, pairs: list[dict]) -> list[Entry]:
    """Notation R: the pairs of a pairs.json in order, a's episode before b's."""
    return [
        (
            {
                "a": clip_path(directory, p["a"]["shard"], p["a"]["id"]),
                "b": clip_path(directory, p["b"]["shard"], p["b"]["id"]),
                "labels": p["labels"],
            },
            (p["a"]["episode"], p["b"]["episode"]),
        )
        for p in pairs
    ]


def shard_files(
    directory: str, file: str, n: int, fingerprint: "hashlib._Hash", update: Callable
) -> Iterator[tuple[int, bytes]]:
    """(I): the file of each of the n complete shards of {directory}/shards/ in order,
    fed to its dataset's fingerprint (4) as read; the bar advanced after each."""
    for k in range(n):
        data = (shard_path(REPO / directory / "shards", k) / file).read_bytes()
        fingerprint.update(data)
        yield k, data
        update()


def read_inputs(
    cfg: "TrainIndexConfig", fields: list[dict]
) -> tuple[list[Entry], list[str | None]]:
    """(I), (4): the shards of C, Q_c, Q_l one at a time (bar `index inputs`); U's then
    R's entries; the fingerprints of C, Q_c (from `fields`), Q_l (None if no pairs)."""
    paired = cfg.pair_clips_dir is not None
    dirs = [
        cfg.clips_dir,
        *([cfg.pair_clips_dir, cfg.pair_labels_dir] if paired else []),
    ]
    hashes = [hashlib.sha256(json.dumps(f).encode()) for f in fields]
    if paired:
        hashes.append(hashlib.sha256())
    n = [complete_shards(REPO / d / "shards") for d in dirs]
    entries = []
    with shard_bar("index inputs", sum(n), 0, sum(n), 0, "shard") as update:
        for k, data in shard_files(dirs[0], RECORDS, n[0], hashes[0], update):
            entries += unlabeled(cfg.clips_dir, k, json.loads(data))
        if paired:
            for _ in shard_files(dirs[1], RECORDS, n[1], hashes[1], update):
                pass
            for _, data in shard_files(dirs[2], PAIRS, n[2], hashes[2], update):
                entries += labeled(cfg.pair_clips_dir, json.loads(data))
    fingerprints = [h.hexdigest() for h in hashes]
    return entries, fingerprints + [None] * (3 - len(fingerprints))


def components(episodes: list[str], edges: list[tuple[str, ...]]) -> list[int]:
    """Notation: per episode in e order, the j of its group G_j: the connected
    components of the pairs' edges, ordered by their first episode."""
    linked = {e: [] for e in episodes}
    for a, b in edges:
        linked[a].append(b)
        linked[b].append(a)
    group, j = {}, 0
    for e in episodes:
        if e in group:
            continue
        group[e], stack = j, [e]
        while stack:
            for f in linked[stack.pop()]:
                if f not in group:
                    group[f] = j
                    stack.append(f)
        j += 1
    return [group[e] for e in episodes]


def counts(items: list[list[dict]], parts: list[int]) -> dict:
    """(W) `counts`: per part, its unlabeled items, labeled items and episodes."""
    return {
        PARTS[x]: {
            "clips": sum("clip" in item for item in items[x]),
            "pairs": sum("clip" not in item for item in items[x]),
            "episodes": parts.count(x),
        }
        for x in (0, 1)
    }


def split(
    cfg: "TrainIndexConfig", entries: list[Entry]
) -> tuple[list[list[dict]], dict]:
    """(1)-(3) with g: one draw per group G_j in order, then sigma_0, sigma_1; per part
    its items I_x in shuffled order (shard s: positions sB .. (s+1)B - 1); counts."""
    episodes = list(dict.fromkeys(e for _, eps in entries for e in eps))
    group = components(episodes, [eps for _, eps in entries if len(eps) == 2])
    g = np.random.Generator(np.random.PCG64([cfg.seed, 2]))
    v = float(cfg.validation)
    n_groups = len(set(group))
    drawn = [
        int(g.choice(2, p=[1 - v, v], replace=True, size=None)) for _ in range(n_groups)
    ]
    part = dict(zip(episodes, [drawn[j] for j in group]))
    items = [[item for item, eps in entries if part[eps[0]] == x] for x in (0, 1)]
    shuffled = [[items_x[i] for i in g.permutation(len(items_x))] for items_x in items]
    return shuffled, counts(items, list(part.values()))
