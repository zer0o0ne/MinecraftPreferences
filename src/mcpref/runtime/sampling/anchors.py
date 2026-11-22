"""Immutable pre-observation memory snapshots; specs/rl-continuous.md."""

import json
from pathlib import Path

import torch

from mcpref.infrastructure.runs.files import sha256, write_json
from mcpref.infrastructure.shards.store import name, shard_path
from mcpref.runtime.sampling.continuous import decode, encode


def initialize(staged: Path, stride: int, fingerprint: str) -> None:
    write_json(
        staged / "anchors.json",
        {"version": 1, "stride": stride, "fingerprint": fingerprint, "files": {}},
    )


def layout(directory: Path, stride: int) -> dict:
    path = directory / "anchors.json"
    if not path.is_file():
        raise RuntimeError("Continuous RL requires sampling memory anchors")
    value = json.loads(path.read_text())
    if value.get("version") != 1 or value.get("stride") != stride:
        raise RuntimeError("incompatible continuous RL memory anchors")
    return value


def write_anchor(staged, index, stream, fingerprint, stride) -> None:
    """Save memory before the observation; publication belongs to its shard."""
    if index % stride:
        return
    manifest = layout(staged, stride)
    path = staged / "anchors" / f"{name(index)}.pt"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise RuntimeError(f"sampling memory anchor already exists: {index}")
    value = {
        "index": index,
        "fingerprint": fingerprint,
        "stream": None if index == 0 else encode(stream),
    }
    torch.save(value, path)
    manifest["files"][str(index)] = sha256(path)
    write_json(staged / "anchors.json", manifest)


def read_anchor(out_dir, index, steps, phi, device, stride):
    """Load a detached anchor and verify its immutable bytes and provenance."""
    owner = max(0, index - 1) // steps
    directory = shard_path(out_dir / "shards", owner)
    manifest = layout(directory, stride)
    path = directory / "anchors" / f"{name(index)}.pt"
    if not path.is_file() or manifest["files"].get(str(index)) != sha256(path):
        raise RuntimeError(f"missing or changed continuous memory anchor: {index}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["index"] != index or value["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError(f"incompatible continuous memory anchor: {index}")
    if index == 0 and value["stream"] is None:
        return None
    if value["stream"] is None:
        raise RuntimeError(f"empty continuous memory anchor: {index}")
    return decode(phi, value["stream"], device).state
