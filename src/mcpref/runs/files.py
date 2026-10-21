"""specs/runs.md: the run files, completeness (1), file hashes (2), JSON files (3),
fingerprints (4); specs/train-index.md: the clip-defining fields, (4) per dataset."""

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from mcpref import REPO
from mcpref.shards.store import complete_shards, shard_path, write_file

HASHED = ("actions", "clips", "screens", "gui_actions", "overlay", "overlay_screens")
CLIP_FIELDS = ("config.width", "config.height", "config.states")
CLIP_FIELDS += (*(f"sha256.{k}" for k in HASHED), "classifier")
ABSENT = object()


def finished_run(directory: Path, written: str, marker: str) -> dict:
    """(1): the parsed {directory}/run.json when its `marker` is neither false nor null,
    else aborts (`written` shown), a missing file included."""
    path = directory / "run.json"
    run = json.loads(path.read_text()) if path.exists() else {}
    value = run.get(marker)
    if value is None or value is False:
        raise RuntimeError(f"{written} is not complete")
    return run


def sha256(path: Path) -> str:
    """(2): SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: object) -> bytes:
    """(3): the bytes of a JSON file, json.dumps(value, indent=2) + newline."""
    return (json.dumps(value, indent=2) + "\n").encode()


def write_json(path: Path, value: dict) -> None:
    """(3): a JSON file written by specs/shards.md (4)."""
    write_file(path, json_bytes(value))


def check_fingerprints(path: Path, fingerprints: dict, out_dir: str) -> None:
    """(4): an existing run file `path` holds `fingerprints` (keys of its `sha256`, else
    top-level keys), compared in order."""
    if path.exists():
        run = json.loads(path.read_text())
        stored = run | run["sha256"]
        for name, value in fingerprints.items():
            if stored[name] != value:
                raise RuntimeError(
                    f"{name} changed since the run started; delete {out_dir}"
                )


def clip_fields(run: dict) -> dict:
    """specs/train-index.md Notation: the clip-defining fields present in run.json,
    {dotted name: value}."""
    fields = {}
    for name in CLIP_FIELDS:
        node = run
        for key in name.split("."):
            node = node.get(key, ABSENT) if isinstance(node, dict) else ABSENT
        if node is not ABSENT:
            fields[name] = node
    return fields


def shard_files(
    directory: str, file: str, n: int, fingerprint: "hashlib._Hash", update: Callable
) -> Iterator[tuple[int, bytes]]:
    """specs/train-index.md (I): `file` of the n complete shards of {directory}/shards/
    in order, fed to the dataset's fingerprint (4) as read; `update()` after each."""
    for k in range(n):
        data = (shard_path(REPO / directory / "shards", k) / file).read_bytes()
        fingerprint.update(data)
        yield k, data
        update()


def new_hash(fields: dict | None) -> "hashlib._Hash":
    """specs/train-index.md (4): the hash before the shards; json.dumps(fields) fed when
    not None."""
    h = hashlib.sha256()
    if fields is not None:
        h.update(json.dumps(fields).encode())
    return h


def fingerprint(
    directory: str, file: str, fields: dict | None, update: Callable[[], object]
) -> str:
    """specs/train-index.md (4) of one dataset: `fields`, then `file` of each complete
    shard of {directory}/shards/, read one at a time, `update()` after each."""
    h = new_hash(fields)
    n = complete_shards(REPO / directory / "shards")
    for _ in shard_files(directory, file, n, h, update):
        pass
    return h.hexdigest()
