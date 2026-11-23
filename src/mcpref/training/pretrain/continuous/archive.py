"""Immutable latent-history shards and checkpointed live-edge references."""

from collections import OrderedDict
from pathlib import Path

import torch

from mcpref.datasets.continuous.config import digest
from mcpref.infrastructure.shards.store import (
    complete_shards,
    make_dirs,
    remove,
    shard_path,
    sync,
    write_shard,
)
from mcpref.training.pretrain.continuous.archive_metadata import Keys, Live, Metadata


class Archive:
    def __init__(self, root: Path, shard_size: int, cache_size: int) -> None:
        self.root, self.shard_size, self.cache_size = root, shard_size, cache_size
        make_dirs(root.parent)
        self.metadata = Metadata(root.parent / "working.sqlite")
        self.live = Live(self.metadata, "live")
        self.consumed = Keys(self.metadata, "consumed")
        self.seen = Keys(self.metadata, "seen")
        self.buffer: dict[str, torch.Tensor] = {}
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.next_shard = 0

    def _cache(self, key: str, value: torch.Tensor) -> torch.Tensor:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def put(self, key: str, history: torch.Tensor, remaining: int) -> None:
        self.metadata.put(key, self.next_shard, remaining)
        if not remaining:
            return
        self.buffer[key] = history.detach().to("cpu", copy=True)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            with write_shard(self.root, self.next_shard) as staged:
                torch.save(self.buffer, staged / "histories.pt")
            self.next_shard += 1
            self.buffer = {}

    def get(self, key: str, like: torch.Tensor) -> torch.Tensor:
        value = self.buffer.get(key)
        if value is None:
            value = self.cache.get(key)
            if value is None:
                path = shard_path(self.root, self.live[key]["shard"]) / "histories.pt"
                shard = torch.load(path, map_location="cpu", weights_only=True)
                value = self._cache(key, shard[key])
            else:
                self.cache.move_to_end(key)
        return value.to(device=like.device, dtype=like.dtype)

    def consume(self, edge: str, other: str) -> None:
        if not self.metadata.consume(edge, other):
            self.cache.pop(other, None)
            # Unpublished retired values need not occupy the next shard.
            self.buffer.pop(other, None)

    def snapshot(self) -> dict:
        self.flush()
        directory = self.root.parent / "metadata"
        k = complete_shards(directory)
        with write_shard(directory, k) as staged:
            self.metadata.snapshot(staged / "state.sqlite")
        return {
            "next_shard": self.next_shard,
            "metadata": {
                "index": k,
                "sha256": digest(shard_path(directory, k) / "state.sqlite"),
            },
        }

    def restore(self, state: dict) -> None:
        self.next_shard = state["next_shard"]
        count = complete_shards(self.root)
        if count < self.next_shard:
            raise RuntimeError("checkpoint references missing latent archive shards")
        for j in range(self.next_shard, count):
            remove(shard_path(self.root, j))
        if self.root.exists():
            sync(self.root)
        directory = self.root.parent / "metadata"
        descriptor = state.get("metadata")
        keep = descriptor["index"] + 1 if descriptor else 0
        if descriptor:
            path = shard_path(directory, descriptor["index"]) / "state.sqlite"
            if digest(path) != descriptor["sha256"]:
                raise RuntimeError("Checkpoint archive metadata checksum differs")
            self.metadata.restore(path)
        elif self.next_shard or any(
            state.get(key) for key in ("live", "seen", "consumed")
        ):
            raise ValueError("Unsupported preference archive checkpoint format")
        else:
            self.metadata.restore()
        for k in range(keep, complete_shards(directory)):
            remove(shard_path(directory, k))
        if directory.exists():
            sync(directory)
        invalid = self.metadata.db.execute(
            "SELECT 1 FROM live WHERE shard<0 OR shard>=? OR remaining<=0 LIMIT 1",
            (self.next_shard,),
        ).fetchone()
        if invalid is not None:
            raise RuntimeError(
                "Checkpoint archive metadata references an unpublished shard"
            )
        for (shard,) in self.metadata.db.execute("SELECT DISTINCT shard FROM live"):
            if not (shard_path(self.root, shard) / "histories.pt").is_file():
                raise RuntimeError(
                    "Checkpoint references missing latent archive payload"
                )
        self.buffer.clear()
        self.cache.clear()

    def close(self):
        self.metadata.close()
