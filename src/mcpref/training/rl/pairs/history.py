"""Storage-backed persistent History of specs/rl-preference-pairs.md."""

import json
import math
import sys
from collections.abc import Iterator, Sequence
from heapq import heapify, heappop, heappush
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import complete_shards
from mcpref.training.rl.pairs.files import (
    load_generation,
    publish_generation,
    read_json_shard,
)
from mcpref.training.rl.pairs.types import (
    Comparison,
    CurrentComparison,
    StateRef,
    Vertex,
    attach,
    read_vertex,
    ref_dict,
    state_ref,
    update,
    vertex_json,
)


class History:
    """One complete generation plus uncommitted insertions, updates and cleanup."""

    def __init__(
        self,
        root: Path,
        history_per_shard: int | None = None,
        config: dict | None = None,
    ) -> None:
        self.root = root
        self.history_per_shard = history_per_shard or 1
        self.config = config or {}
        self.generation = -1
        self._new: dict[StateRef, Vertex] = {}
        self._updates: dict[StateRef, Vertex] = {}
        self._removed: set[StateRef] = set()
        self._compared: set[StateRef] = set()
        self._aged = False
        self._base_index: dict[StateRef, tuple[int, int]] | None = None
        self._base_cache: tuple[int, list[Vertex]] | None = None

    @classmethod
    def load(
        cls,
        root: Path,
        history_per_shard: int | None = None,
        config: dict | None = None,
    ) -> "History":
        """Load the newest complete generation; discard an incomplete staged one."""
        loaded = load_generation(root)
        if loaded is None:
            return cls(root, history_per_shard, config)
        cycle, run = loaded
        per_shard = history_per_shard or run["config"]["history_per_shard"]
        history = cls(root, per_shard, config or run["config"])
        history.generation = cycle
        return history

    @property
    def base(self) -> Path:
        """The loaded complete generation, absent for an empty History."""
        return self.root / f"{self.generation:06d}"

    def _base_shard(self, k: int) -> list[Vertex]:
        if self._base_cache is not None and self._base_cache[0] == k:
            return self._base_cache[1]
        entries = read_json_shard(self.base / "shards", k, "vertices.json")
        vertices = [read_vertex(entry) for entry in entries]
        self._base_cache = (k, vertices)
        return vertices

    def _base_vertices(self) -> Iterator[Vertex]:
        if self.generation < 0:
            return
        for k in range(complete_shards(self.base / "shards")):
            yield from self._base_shard(k)

    def _base_vertex(self, ref: StateRef) -> Vertex | None:
        if self.generation < 0:
            return None
        if self._base_index is None:
            index = {}
            for k in range(complete_shards(self.base / "shards")):
                for offset, vertex in enumerate(self._base_shard(k)):
                    index[vertex.ref] = (k, offset)
            self._base_index = index
        position = self._base_index.get(ref)
        if position is None:
            return None
        shard, offset = position
        return self._base_shard(shard)[offset]

    def _raw(self, ref: StateRef) -> Vertex | None:
        if ref in self._updates:
            return self._updates[ref]
        if ref in self._new:
            return self._new[ref]
        return self._base_vertex(ref)

    def _effective(self, v: Vertex) -> Vertex:
        unused = v.unused
        if self._aged:
            unused = 0 if v.ref in self._compared else unused + 1
        comparisons = tuple(c for c in v.comparisons if c.ref not in self._removed)
        return Vertex(v.ref, v.key, unused, comparisons)

    def _vertices(self) -> Iterator[Vertex]:
        old = iter(self._base_vertices())
        new = iter(sorted(self._new.values(), key=lambda v: v.ref))
        a, b = next(old, None), next(new, None)
        while a is not None or b is not None:
            if b is None or (a is not None and a.ref < b.ref):
                v, a = a, next(old, None)
            else:
                v, b = b, next(new, None)
            if v.ref not in self._removed:
                yield self._effective(self._updates.get(v.ref, v))

    def get(self, ref: StateRef) -> Vertex | None:
        """Return the staged view of one vertex, or None after eviction/if absent."""
        if ref in self._removed:
            return None
        vertex = self._raw(ref)
        return None if vertex is None else self._effective(vertex)

    def older_comparison(self, a: StateRef, b: StateRef) -> bool:
        """Whether the loaded generation already contains this unordered comparison."""
        vertex = self._base_vertex(a)
        return vertex is not None and any(c.ref == b for c in vertex.comparisons)

    def outside(self, episode: str) -> Iterator[Vertex]:
        """Iterate vertices outside an episode while reading one shard at a time."""
        for vertex in self._vertices():
            if vertex.ref.episode != episode:
                yield vertex

    def outside_shards(self, episode: str) -> Iterator[list[Vertex]]:
        """Iterate the staged outside view in shard-bounded groups."""
        vertices = self.outside(episode)
        while batch := list(islice(vertices, self.history_per_shard)):
            yield batch

    def stage_seed(self, ref: StateRef, key: Sequence[float]) -> None:
        """Stage Eq. (2)'s empty seed vertex, preserving an existing insertion."""
        if self._raw(ref) is None:
            self._new[ref] = Vertex(ref, tuple(key), 0, ())

    def replace_key(self, ref: StateRef, key: Sequence[float]) -> None:
        """Eq. (12): replace only a fresh seed's provisional selection key."""
        if ref not in self._new:
            raise KeyError("final insertion keys are limited to fresh selected seeds")
        vertex = self._updates.get(ref, self._new[ref])
        self._updates[ref] = Vertex(ref, tuple(key), vertex.unused, vertex.comparisons)

    def stage_comparison(self, c: CurrentComparison) -> None:
        """Stage Eq. (7)'s symmetric mean updates and actual-comparison usage."""
        a, b = self.get(c.a), self.get(c.b)
        if a is None or b is None:
            raise KeyError("current comparison endpoints must be History vertices")
        forward = update(a, c.b, c.p, c.inc)
        record = next(record for record in forward.comparisons if record.ref == c.b)
        reverse = Comparison(c.a, 1 - record.p, record.inc, record.n)
        self._updates[c.a] = forward
        self._updates[c.b] = attach(b, reverse)
        self._compared.update((c.a, c.b))

    def age_unused(self) -> None:
        """Stage Eq. (7)'s reset/aging for the end of this sampling cycle."""
        self._aged = True

    @staticmethod
    def _cleanup_entry(
        ref: StateRef, usage: dict[StateRef, tuple[int, int]], version: int
    ) -> tuple[float, int, StateRef, int]:
        unused, n = usage[ref]
        return (math.sqrt(n / (unused + 1)), -unused, ref, version)

    def _cleanup_state(
        self,
    ) -> tuple[
        list[tuple[float, int, StateRef, int]],
        dict[StateRef, tuple[int, int]],
        dict[StateRef, int],
        dict[StateRef, dict[StateRef, int]],
    ]:
        usage, versions, reverse, heap = {}, {}, {}, []
        for v in self._vertices():
            usage[v.ref] = (v.unused, v.n)
            versions[v.ref] = 0
            heap.append(self._cleanup_entry(v.ref, usage, 0))
            for c in v.comparisons:
                reverse.setdefault(c.ref, {})[v.ref] = c.n
        heapify(heap)
        return heap, usage, versions, reverse

    def _evict(
        self,
        heap: list[tuple[float, int, StateRef, int]],
        usage: dict[StateRef, tuple[int, int]],
        versions: dict[StateRef, int],
        reverse: dict[StateRef, dict[StateRef, int]],
    ) -> StateRef:
        """Remove the next vertex by Eq. (10)'s score and stable tie breaks."""
        while heap:
            _, _, ref, version = heappop(heap)
            if ref not in self._removed and versions[ref] == version:
                break
        else:
            raise RuntimeError("History cleanup has no vertex to remove")
        self._removed.add(ref)
        for neighbor, removed_n in reverse.get(ref, {}).items():
            if neighbor in self._removed:
                continue
            unused, n = usage[neighbor]
            usage[neighbor] = (unused, n - removed_n)
            versions[neighbor] += 1
            entry = self._cleanup_entry(neighbor, usage, versions[neighbor])
            heappush(heap, entry)
        return ref

    def cleanup(self, capacity: int, progress_path: Path | None = None) -> None:
        """Eq. (10): deterministic removals, checkpointed after each removal."""
        total = max(sum(1 for _ in self._vertices()) - capacity, 0)
        restored: list[StateRef] = []
        if progress_path is not None and progress_path.exists():
            saved = json.loads(progress_path.read_text())
            restored = [state_ref(r) for r in saved["removed"]]
            self._removed.update(restored)
        initial = len(restored)
        if total == 0:
            print("rl pair cleanup: nothing to do", file=sys.stderr)
            return
        if initial == total:
            print("rl pair cleanup: already done", file=sys.stderr)
            return
        heap, usage, versions, reverse = self._cleanup_state()
        with tqdm(
            total=total, initial=initial, desc="rl pair cleanup", unit="vertex"
        ) as bar:
            for _ in range(total - initial):
                restored.append(self._evict(heap, usage, versions, reverse))
                if progress_path is not None:
                    write_json(
                        progress_path,
                        {"removed": [ref_dict(r) for r in restored]},
                    )
                bar.update(1)

    def publish(self, cycle: int) -> Path:
        """Publish all staged work as the only complete History generation."""
        vertices = (vertex_json(vertex) for vertex in self._vertices())
        return publish_generation(
            self.root, cycle, vertices, self.history_per_shard, self.config
        )
