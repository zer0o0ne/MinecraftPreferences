"""specs/runs.md: the run files of the blocks; the completeness of another block's output
(1), file hashes (2), JSON files (3), fingerprints (4)."""

import hashlib
import json
from pathlib import Path

from mcpref.shards.store import write_file


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


def write_json(path: Path, value: dict) -> None:
    """(3): a JSON file written by specs/shards.md (4)."""
    write_file(path, (json.dumps(value, indent=2) + "\n").encode())


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
