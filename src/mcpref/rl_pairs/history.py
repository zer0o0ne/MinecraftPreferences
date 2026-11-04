"""Storage-backed persistent History of specs/rl-preference-pairs.md."""

import json
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from tqdm import tqdm

from mcpref.rl_pairs.files import (
    load_generation,
    publish_generation,
    read_json_shard,
)
from mcpref.rl_pairs.types import (
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
from mcpref.runs.files import write_json
from mcpref.shards.store import complete_shards


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

    def _base_vertices(self) -> Iterator[Vertex]:
        if self.generation < 0:
            return
        shards = self.base / "shards"
        for k in range(complete_shards(shards)):
            entries = read_json_shard(shards, k, "vertices.json")
            for entry in entries:
                yield read_vertex(entry)

    def _raw(self, ref: StateRef) -> Vertex | None:
        if ref in self._updates:
            return self._updates[ref]
        if ref in self._new:
            return self._new[ref]
        for vertex in self._base_vertices():
            if vertex.ref == ref:
                return vertex
        return None

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

    def outside(self, episode: str) -> Iterator[Vertex]:
        """Iterate vertices outside an episode while reading one shard at a time."""
        for vertex in self._vertices():
            if vertex.ref.episode != episode:
                yield vertex

    def stage_seed(self, ref: StateRef, key: Sequence[float]) -> None:
        """Stage Eq. (2)'s empty seed vertex, preserving an existing insertion."""
        if self._raw(ref) is None:
            self._new[ref] = Vertex(ref, tuple(key), 0, ())

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

    def _evict(self) -> StateRef:
        """Remove the next vertex by Eq. (10)'s score and stable tie breaks."""
        best: Vertex | None = None
        best_key = None
        for v in self._vertices():
            key = (math.sqrt(v.n / (v.unused + 1)), -v.unused, v.ref)
            if best_key is None or key < best_key:
                best, best_key = v, key
        if best is None:
            raise RuntimeError("History cleanup has no vertex to remove")
        self._removed.add(best.ref)
        return best.ref

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
        with tqdm(
            total=total, initial=initial, desc="rl pair cleanup", unit="vertex"
        ) as bar:
            for _ in range(total - initial):
                restored.append(self._evict())
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
