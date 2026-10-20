"""Transcription of specs/train-index.md for its tests, written from the spec alone: the
items of Notation ($U$, $R$, the episodes, the groups), (1)-(3) with
`Generator(PCG64([seed, 2]))` ((1) as [N] reads `choice`: one `random()` u per group, part
1 iff u >= 1 - v), the fingerprints (4) and the `run.json` of (R') and (W)."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

HASHED = ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens")
CLIP_FIELDS = (  # Notation: the clip-defining fields, in their order
    "config.width",
    "config.height",
    "config.states",
    *(f"sha256.{k}" for k in HASHED),
    "classifier",
)
PARTS = ("train", "validation")


def name(k: int) -> str:
    """specs/shards.md: name(k)."""
    return f"{k:06d}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def field(run: dict, dotted: str):
    """The value of a clip-defining field in a `run.json`."""
    value = run
    for key in dotted.split("."):
        value = value[key]
    return value


def clips_fingerprint(run: dict, records: list[bytes]) -> str:
    """(4) of a clips dataset: one json.dumps of {dotted name: value} (default separators),
    then the bytes of each complete shard's records.json in shard order."""
    head = json.dumps({f: field(run, f) for f in CLIP_FIELDS}).encode("utf-8")
    return sha(head + b"".join(records))


def labels_fingerprint(pairs: list[bytes]) -> str:
    """(4) of the labels: the bytes of each complete shard's pairs.json in shard order."""
    return sha(b"".join(pairs))


def shard_files(directory: Path, file: str) -> list[bytes]:
    """The bytes of `file` in the shards 000000, 000001, ... of `directory` (the staged
    ones ignored), in shard order."""
    n = 0
    while (directory / name(n)).is_dir():
        n += 1
    return [(directory / name(k) / file).read_bytes() for k in range(n)]


def clip_path(written: str, shard: int, clip: str) -> str:
    """Notation, Item: p of a clip of a clips dataset whose config path string is `written`."""
    return str(PurePosixPath(written) / "shards" / name(shard) / "clips" / clip)


def unlabeled(clips: list[list[dict]], written: str) -> list[tuple[dict, str]]:
    """$U$ from the records of C's complete shards: (item, episode) in order."""
    return [
        ({"clip": clip_path(written, k, i)}, r["relpath"])
        for k, records in enumerate(clips)
        for r in records
        if r["clips"] is not None
        for i in r["clips"]
    ]


def labeled(pairs: list[list[dict]], written: str) -> list[tuple[dict, str, str]]:
    """$R$ from the pairs of Q_l's complete shards: (item, episode of a, of b) in order."""
    out = []
    for shard in pairs:
        for p in shard:
            a, b = p["a"], p["b"]
            item = {"a": clip_path(written, a["shard"], a["id"])}
            item |= {
                "b": clip_path(written, b["shard"], b["id"]),
                "labels": p["labels"],
            }
            out.append((item, a["episode"], b["episode"]))
    return out


def episodes(u: list, r: list) -> list[str]:
    """Notation: in order of first appearance over U, then over R, a before b."""
    seen = [e for _, e in u] + [e for _, a, b in r for e in (a, b)]
    return list(dict.fromkeys(seen))


def groups(eps: list[str], r: list) -> dict[str, int]:
    """episode -> j of its group G_j: connected components of the pairs' graph, ordered by
    their first episode in the order of e (a naive flood fill)."""
    edges = {e: set() for e in eps}
    for _, a, b in r:
        edges[a].add(b)
        edges[b].add(a)
    group: dict[str, int] = {}
    for e in eps:
        if e in group:
            continue
        j, todo = len(set(group.values())), [e]
        while todo:
            x = todo.pop()
            if x not in group:
                group[x] = j
                todo.extend(edges[x])
    return group


@dataclass(frozen=True)
class Expected:
    shards: tuple[list, list]  # per part: the shards, each a list of items
    counts: dict  # counts of (W)
    parts: list[int]  # part of each group G_j


def expected(clips: list, pairs: list | None, config: dict) -> Expected:
    """(1)-(3) and the counts of (W) on the records of C's and the pairs of Q_l's complete
    shards (pairs None: no pairs)."""
    u = unlabeled(clips, config["clips_dir"])
    r = labeled(pairs, config["pair_clips_dir"]) if pairs is not None else []
    eps = episodes(u, r)
    group = groups(eps, r)
    g = np.random.Generator(np.random.PCG64([config["seed"], 2]))
    v = float(config["validation"])
    parts = [int(g.random() >= 1 - v) for _ in range(len(set(group.values())))]
    items: tuple[list, list] = ([], [])
    for item, e in u:
        items[parts[group[e]]].append(item)
    for item, a, b in r:
        assert group[a] == group[b]
        items[parts[group[a]]].append(item)
    sigma = [g.permutation(len(items[0])), g.permutation(len(items[1]))]
    size = config["items_per_shard"]
    shards = tuple(
        [
            [items[x][i] for i in sigma[x][s : s + size]]
            for s in range(0, len(items[x]), size)
        ]
        for x in (0, 1)
    )
    counts = {
        part: {
            "clips": sum(parts[group[e]] == x for _, e in u),
            "pairs": sum(parts[group[a]] == x for _, a, _ in r),
            "episodes": sum(parts[group[e]] == x for e in eps),
        }
        for x, part in enumerate(PARTS)
    }
    return Expected(shards, counts, parts)


def items_bytes(items: list) -> bytes:
    """(W): items.json of a shard."""
    return (json.dumps({"items": items}, indent=2) + "\n").encode()


def run_json(config_bytes: bytes, fingerprints: dict, clips_run: dict) -> dict:
    """(R'): {out_dir}/run.json; fingerprints: clips, pair_clips, pair_labels."""
    sha256 = {"config": sha(config_bytes), **fingerprints}
    return {
        "config": yaml.safe_load(config_bytes),
        "sha256": sha256,
        "states": clips_run["config"]["states"],
        "width": clips_run["config"]["width"],
        "height": clips_run["config"]["height"],
    }


def json_bytes(value) -> bytes:
    """write_json: json.dumps(value, indent=2) + "\\n"."""
    return (json.dumps(value, indent=2) + "\n").encode()
