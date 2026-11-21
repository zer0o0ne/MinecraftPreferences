"""The unique-clip objective and its training-index compatibility boundary."""

import json
from pathlib import Path

from mcpref.infrastructure.shards.store import complete_shards

INDEX_SEMANTICS = "unique-clips-v1"


def validate_index(run: dict, name: str) -> None:
    """Reject indexes built for occurrence-weighted clip objectives."""
    if run.get("semantics") != INDEX_SEMANTICS or not run.get("sha256", {}).get(
        "clip_metadata"
    ):
        raise RuntimeError(f"{name} must be rebuilt for {INDEX_SEMANTICS}")


def validate_destination(directory: Path, name: str) -> None:
    """Complete shards may only be resumed with a compatible run header."""
    path = directory / "run.json"
    if path.exists():
        validate_index(json.loads(path.read_bytes()), name)
    elif any(complete_shards(directory / part) for part in ("train", "validation")):
        raise RuntimeError(f"{name} has index shards without run.json; rebuild it")
