"""Configuration and frozen acquisition order for continuous-data.md."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from mcpref.config import config_mapping, positive_int, read_yaml, relative


@dataclass(frozen=True)
class ContinuousConfig:
    perception_config: str
    acquisition_plan: str
    legacy_provenance: str
    pair_clips_dir: str
    pair_labels_dir: str
    split_config: str
    source_cache: str
    out_dir: str
    shard_states: int
    metadata_rows: int
    max_shift: int
    join_gap_ms: int
    disk_reserve_bytes: int
    codec_threads: int


def load_config(path: Path) -> ContinuousConfig:
    raw = config_mapping(read_yaml(path), path, ContinuousConfig)
    for key in ContinuousConfig.__annotations__:
        if ContinuousConfig.__annotations__[key] is str:
            relative(key, raw[key])
        else:
            positive_int(key, raw[key])
    return ContinuousConfig(**raw)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_jsonl(path: Path, expected: str):
    if digest(path) != expected:
        raise ValueError(f"Frozen manifest changed: {path}")
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def plans(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    shards = manifest["shards"]
    pieces = itertools.chain.from_iterable(
        checked_jsonl(root / s["path"], s["sha256"]) for s in shards
    )
    file = manifest["fragments_file"]
    fragments = checked_jsonl(root / file["path"], file["sha256"])
    seen, total = set(), 0
    groups = itertools.groupby(pieces, key=lambda p: p["fragment_id"])
    for k, ((identity, group), fragment) in enumerate(
        zip(groups, fragments, strict=True)
    ):
        group = list(group)
        if (
            identity != fragment["fragment_id"]
            or [p["piece_id"] for p in group] != fragment["piece_ids"]
        ):
            raise ValueError("Fragment catalogue and source order differ")
        for j, piece in enumerate(group):
            if piece["piece_id"] in seen or piece["fragment_position"] != j:
                raise ValueError("Repeated or out-of-order source")
            seen.add(piece["piece_id"])
        total += len(group)
        yield k, fragment, group
    if total != manifest["totals"]["pieces"]:
        raise ValueError("Manifest source count differs")
