"""Atomic output shards and exact recovery of a partially published fragment."""

import hashlib
import json
from pathlib import Path

from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.frames import encode
from mcpref.datasets.continuous.sources import reserve
from mcpref.datasets.continuous.timeline import observations
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import shard_path, write_shard


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows
    )


def publish_batch(
    pipe, cfg, out: Path, index: int, raw: Path, items: list, intervals: list
) -> dict:
    observation_bytes, interval_bytes = jsonl(items), jsonl(intervals)
    state = {
        "fragment_id": items[-1]["fragment_id"],
        "last_observation": items[-1],
        "observations": len(items),
        "intervals": len(intervals),
        "observation_sha256": hashlib.sha256(observation_bytes).hexdigest(),
        "interval_sha256": hashlib.sha256(interval_bytes).hexdigest(),
        "raw_sha256": digest(raw),
    }
    target = shard_path(out / "shards", index)
    if target.exists():
        previous = json.loads((target / "state.json").read_text())
        if any(previous[k] != v for k, v in state.items()):
            raise ValueError(f"Resumed fragment differs from published shard {index}")
        for name, checksum in previous["files"].items():
            if digest(target / name) != checksum:
                raise ValueError(f"Published shard checksum differs: {target / name}")
        return previous
    reserve(out, raw.stat().st_size, cfg.disk_reserve_bytes)
    with write_shard(out / "shards", index) as staged:
        (staged / "observations.jsonl").write_bytes(observation_bytes)
        (staged / "intervals.jsonl").write_bytes(interval_bytes)
        codec = encode(pipe.cfg, cfg, raw, staged / "frames.mkv", len(items))
        files = {
            name: digest(staged / name)
            for name in ("observations.jsonl", "intervals.jsonl", "frames.mkv")
        }
        state |= {
            "codec": codec,
            "files": files,
            "frame_payload_bytes": (staged / "frames.mkv").stat().st_size,
        }
        (staged / "state.json").write_bytes(json_bytes(state))
    return state


def write_timeline(
    pipe, cfg, out: Path, timeline: dict, sources: list, index: int
) -> dict:
    raw = out / "staging" / "frames.yuv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    items, intervals, previous = [], [], None
    counts = {
        "observations": 0,
        "intervals": 0,
        "valid_intervals": 0,
        "frame_payload_bytes": 0,
    }
    first_shard = index
    with raw.open("wb") as stream:
        for item, frame in observations(pipe, timeline, sources):
            if (
                previous is not None
                and previous["grid_index"] + 1 == item["grid_index"]
            ):
                interval = timeline["intervals"][previous["grid_index"]] | {
                    "left": previous["id"],
                    "right": item["id"],
                }
                intervals.append(interval)
                counts["valid_intervals"] += interval["valid"]
            timeline["locations"][item["kept_ordinal"]] = [index, len(items)]
            stream.write(frame)
            items.append(item)
            previous = item
            if len(items) == cfg.shard_states:
                stream.flush()
                state = publish_batch(pipe, cfg, out, index, raw, items, intervals)
                counts["observations"] += len(items)
                counts["intervals"] += len(intervals)
                counts["frame_payload_bytes"] += state["frame_payload_bytes"]
                index += 1
                items, intervals = [], []
                stream.seek(0)
                stream.truncate()
        if items:
            stream.flush()
            state = publish_batch(pipe, cfg, out, index, raw, items, intervals)
            counts["observations"] += len(items)
            counts["intervals"] += len(intervals)
            counts["frame_payload_bytes"] += state["frame_payload_bytes"]
            index += 1
    raw.unlink()
    return counts | {"first_shard": first_shard, "next_shard": index}
