"""Retain recent latent shards and every live/checkpoint archive dependency."""

import json
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing

import torch
from tqdm import tqdm

from mcpref.datasets.continuous.config import digest
from mcpref.infrastructure.shards.store import complete_shards, index, shard_path, sync


def checkpoint_archives(root):
    for path in sorted((root / "checkpoints").glob("*/checkpoint.pt")):
        if index(path.parent.name) is None:
            continue
        header = json.loads(path.with_name("metadata.json").read_text())
        if "archive" in header:
            yield header["epoch"], header["archive"]
        else:
            # Legacy headers predate lightweight archive references.
            state = torch.load(path, map_location="cpu", weights_only=False)
            yield state["position"]["epoch"], state["archive"]


def protect(directory, db, next_shard, protected):
    if db.execute("SELECT 1 FROM live WHERE remaining<=0 LIMIT 1").fetchone():
        raise RuntimeError("Archive snapshot contains an invalid live history")
    for (shard,) in db.execute("SELECT DISTINCT shard FROM live"):
        payload = shard_path(directory / "shards", shard) / "histories.pt"
        if not 0 <= shard < next_shard or not payload.is_file():
            raise RuntimeError("Checkpoint references missing latent archive payload")
        protected.add(shard)


def dependencies(root, archive):
    protected, snapshots = defaultdict(set), defaultdict(set)
    protect(
        archive.root.parent,
        archive.metadata.db,
        archive.next_shard,
        protected[archive.root.parent],
    )
    for epoch, state in checkpoint_archives(root):
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("Invalid checkpoint archive epoch")
        directory = root / "archive" / f"epoch-{epoch}"
        descriptor = state["metadata"]
        path = shard_path(directory / "metadata", descriptor["index"]) / "state.sqlite"
        if digest(path) != descriptor["sha256"]:
            raise RuntimeError("Checkpoint archive metadata checksum differs")
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            protect(directory, db, state["next_shard"], protected[directory])
        snapshots[directory].add(descriptor["index"])
    return protected, snapshots


def obsolete(root, archive, keep):
    protected, snapshots = dependencies(root, archive)
    paths = []
    for directory in sorted((root / "archive").glob("epoch-*")):
        count = complete_shards(directory / "shards")
        if directory == archive.root.parent:
            protected[directory].update(range(max(0, count - keep), count))
        for path in sorted(directory.glob("shards/*/histories.pt")):
            shard = index(path.parent.name)
            if shard is not None and shard not in protected[directory]:
                paths.append(path)
        for path in sorted(directory.glob("metadata/*/state.sqlite")):
            snapshot = index(path.parent.name)
            if snapshot is not None and snapshot not in snapshots[directory]:
                paths.append(path)
        working = directory / "working.sqlite"
        if directory != archive.root.parent and working.exists():
            paths.append(working)
    return paths


def collect(root, archive, keep):
    """Collect only after checkpoint publication/pruning; validate before unlinking."""
    paths = obsolete(root, archive, keep)
    if not paths:
        print("Archive GC: nothing to remove", file=sys.stderr)
        return
    for path in tqdm(paths, total=len(paths), desc="Archive GC", unit="file"):
        path.unlink()
        sync(path.parent)
