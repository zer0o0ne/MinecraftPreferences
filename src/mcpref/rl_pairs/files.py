"""JSON shards and complete History generations of specs/rl-preference-pairs.md."""

import bisect
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from mcpref.runs.files import json_bytes, write_json
from mcpref.shards.store import (
    complete_shards,
    make_dirs,
    shard_path,
    sync,
    write_shard,
)


def read_json_shard(directory: Path, k: int, filename: str) -> object:
    """Read one JSON value from complete shard k."""
    return json.loads((shard_path(directory, k) / filename).read_text())


def write_json_shard(directory: Path, k: int, filename: str, value: object) -> None:
    """Publish one JSON value as complete shard k."""
    with write_shard(directory, k) as path:
        (path / filename).write_bytes(json_bytes(value))


def pair_counts(pairs_dir: Path) -> list[int]:
    """Shard sizes only; no preference dataset is materialised."""
    return [
        len(read_json_shard(pairs_dir / "shards", k, "items.json"))
        for k in range(complete_shards(pairs_dir / "shards"))
    ]


def pair_reader(pairs_dir: Path, counts: list[int]):
    """Resolve a preference source position while retaining at most one pair shard."""
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    cached_shard, cached_items = None, None

    def read(position: int) -> dict:
        nonlocal cached_shard, cached_items
        k = bisect.bisect_right(starts, position) - 1
        if k != cached_shard:
            cached_items = read_json_shard(pairs_dir / "shards", k, "items.json")
            cached_shard = k
        return cached_items[position - starts[k]]

    return read


def shard_items(directory: Path, filename: str, shards: int, per_shard: int) -> int:
    """The item count of complete in-order equally sized shards and their last shard."""
    if shards == 0:
        return 0
    last = read_json_shard(directory, shards - 1, filename)
    return (shards - 1) * per_shard + len(last)


def generations(root: Path) -> list[tuple[int, Path, Path]]:
    """The complete numeric History generations in a root directory."""
    found = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.isascii() and path.name.isdigit():
                run_path = path / "run.json"
                if run_path.exists() and json.loads(run_path.read_text()).get(
                    "complete"
                ):
                    found.append((int(path.name), path, run_path))
    return found


def load_generation(root: Path) -> tuple[int, dict] | None:
    """Discard staged/old generations and read the newest complete run file."""
    if root.exists():
        for path in root.glob("*.part"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    found = generations(root)
    if not found:
        return None
    cycle, path, run_path = max(found, key=lambda item: item[0])
    for _, old, _ in found:
        if old != path:
            shutil.rmtree(old)
    if len(found) > 1:
        sync(root)
    return cycle, json.loads(run_path.read_text())


def write_vertices(staged: Path, vertices: Iterable[dict], per_shard: int) -> None:
    """Write the sorted vertex stream into bounded complete shards."""
    buffer = []
    shard = 0
    for vertex in vertices:
        buffer.append(vertex)
        if len(buffer) == per_shard:
            write_json_shard(staged / "shards", shard, "vertices.json", buffer)
            shard += 1
            buffer = []
    if buffer:
        write_json_shard(staged / "shards", shard, "vertices.json", buffer)


def publish_generation(
    root: Path,
    cycle: int,
    vertices: Iterable[dict],
    per_shard: int,
    config: dict,
) -> Path:
    """Atomically publish a complete generation, then delete every older one."""
    target = root / f"{cycle:06d}"
    staged = Path(f"{target}.part")
    if staged.exists():
        shutil.rmtree(staged)
    make_dirs(staged)
    write_vertices(staged, vertices, per_shard)
    stored_config = config | {"history_per_shard": per_shard}
    write_json(
        staged / "run.json",
        {"config": stored_config, "cycle": cycle, "complete": True},
    )
    make_dirs(root)
    os.rename(staged, target)
    sync(root)
    for path in root.iterdir():
        if path != target and path.is_dir() and path.name.isdigit():
            shutil.rmtree(path)
    sync(root)
    return target
