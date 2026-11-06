"""Eq. (1), C3: disk-backed global order, mixed index and pending tail."""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref import REPO
from mcpref.ppo.config import load_config as load_ppo_config
from mcpref.rl.config import RLCycleConfig
from mcpref.rl.files import incoming_tail, shards_digest, write_tail
from mcpref.rl_pairs.files import (
    pair_counts,
    pair_reader,
    read_json_shard,
    write_json_shard,
)
from mcpref.runs.files import write_json
from mcpref.sampling.config import load_config as load_sampling_config
from mcpref.shards.store import complete_shards, make_dirs
from mcpref.training.schedule import n_batches


def shuffled_order(root: Path, cfg: RLCycleConfig, count: int):
    """Restart an uncommitted shuffle, or reopen only its committed memmap state."""
    directory = root / "order"
    path, state_path = directory / "order.dat", directory / "shuffle.json"
    if (directory / "run.json").exists():
        return None
    if not path.exists() or not state_path.exists():
        if complete_shards(directory / "shards"):
            raise RuntimeError(
                "partial order shards require their memmap shuffle state"
            )
        make_dirs(directory)
        order = np.memmap(path, dtype=np.int64, mode="w+", shape=(count,))
        for start in range(0, count, cfg.items_per_shard):
            stop = min(start + cfg.items_per_shard, count)
            order[start:stop] = np.arange(start, stop, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64([cfg.seed, 2]))
        rng.shuffle(order, axis=0)
        order.flush()
        write_json(state_path, {"rng": rng.bit_generator.state})
        return order
    return np.memmap(path, dtype=np.int64, mode="r+", shape=(count,))


def source_item(
    position: int,
    u_before: int,
    fresh: int,
    pending: list[int],
    read_pair,
) -> dict:
    """Resolve fresh -> pending -> preference source order without a source list."""
    if position < fresh:
        return {"kind": "fresh", "index": u_before + position}
    if position < fresh + len(pending):
        return {"kind": "pending", "index": pending[position - fresh]}
    item = read_pair(position - fresh - len(pending))
    return {
        "kind": "preference",
        "a": item["a"],
        "b": item["b"],
        "labels": item["labels"],
    }


def build_index(cfg: RLCycleConfig, cycle: int, u_before: int, pairs_dir: Path) -> Path:
    """Build resumable order/index shards and the remaining tail exactly once."""
    root = REPO / cfg.out_dir / f"{cycle:06d}"
    index_dir = root / "index"
    if all((root / name / "run.json").exists() for name in ("order", "index", "tail")):
        print("rl index: already done", file=sys.stderr)
        return index_dir
    sampling = load_sampling_config(REPO / cfg.sampling)
    ppo = load_ppo_config(REPO / cfg.ppo)
    pending_all = incoming_tail(root, cycle, u_before)
    u_after = u_before + sampling.steps
    pending = [t for t in pending_all if t + ppo.horizon <= u_after]
    counts = pair_counts(pairs_dir)
    preference, fresh = sum(counts), sampling.steps
    total = fresh + len(pending) + preference
    shards = (total + cfg.items_per_shard - 1) // cfg.items_per_shard
    order_done = complete_shards(root / "order" / "shards")
    index_done = complete_shards(index_dir / "shards")
    exchanges = max(total - 1, 0)
    shuffle_done = (root / "order" / "run.json").exists() or (
        (root / "order" / "order.dat").exists()
        and (root / "order" / "shuffle.json").exists()
    )
    resolved = min(index_done * cfg.items_per_shard, total)
    initial = (exchanges if shuffle_done else 0) + resolved
    work = exchanges + total
    if initial == work:
        print("rl index: already done", file=sys.stderr)
    bar = None
    if initial < work:
        bar = tqdm(
            total=work,
            initial=initial,
            desc="rl index",
            unit="item",
            file=sys.stderr,
        )
    try:
        order = shuffled_order(root, cfg, total)
        if not shuffle_done:
            for _ in range(exchanges):
                if bar is not None:
                    bar.update()
        read_pair = pair_reader(pairs_dir, counts)
        for k in range(shards):
            start = k * cfg.items_per_shard
            stop = min((k + 1) * cfg.items_per_shard, total)
            if k == order_done:
                if order is None:
                    raise RuntimeError(
                        "missing order memmap before all order shards complete"
                    )
                write_json_shard(
                    root / "order" / "shards",
                    k,
                    "indices.json",
                    {"indices": order[start:stop].tolist()},
                )
                order_done += 1
            if k == index_done:
                if order is None:
                    positions = read_json_shard(
                        root / "order" / "shards", k, "indices.json"
                    )["indices"]
                else:
                    positions = order[start:stop].tolist()
                items = []
                for position in positions:
                    items.append(
                        source_item(position, u_before, fresh, pending, read_pair)
                    )
                    if bar is not None:
                        bar.update()
                write_json_shard(
                    index_dir / "shards", k, "items.json", {"items": items}
                )
                index_done += 1
    finally:
        if bar is not None:
            bar.close()
    order_run = root / "order" / "run.json"
    if not order_run.exists():
        rng = json.loads((root / "order" / "shuffle.json").read_text())["rng"]
        write_json(
            order_run,
            {
                "complete": True,
                "rng": rng,
                "sha256": shards_digest(root / "order" / "shards", "indices.json"),
            },
        )
    if order is not None:
        del order
    for path in (root / "order" / "order.dat", root / "order" / "shuffle.json"):
        if path.exists():
            path.unlink()
    write_tail(root, u_after, ppo.horizon)
    sizes = [
        len(read_json_shard(index_dir / "shards", k, "items.json")["items"])
        for k in range(shards)
    ]
    eligible = sum(t + ppo.horizon <= u_after for t in range(u_before, u_after))
    batches = n_batches(sizes, cfg.batch)
    write_json(
        index_dir / "run.json",
        {
            "complete": True,
            "counts": {
                "fresh": fresh,
                "pending": len(pending),
                "ppo": len(pending) + eligible,
                "preference": preference,
                "items": total,
            },
            "batches": batches,
            "steps": cfg.epochs * batches,
            "sha256": shards_digest(index_dir / "shards", "items.json"),
        },
    )
    return index_dir


def read_item(index_dir: Path, shard: int, position: int) -> dict:
    """Read one item while loading only its containing index shard."""
    return read_json_shard(index_dir / "shards", shard, "items.json")["items"][position]
