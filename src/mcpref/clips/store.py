"""Interface Datasets of specs/perception-data.md: records, staged clips, the shards of
(6) and the continuation (7)."""

import json
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from mcpref.clips.sample import Clip
from mcpref.shards.store import complete_shards, shard_path, write_shard
from mcpref.vpt.episodes import episode_name, version_of
from mcpref.vpt.plan import generator

Pools = dict[str, tuple[str, list[str]]]
RECORD = ("subset", "relpath", "reason", "options", "starts_session")
RECORD += ("overlay_start", "valid_starts", "clips")


def json_bytes(value: object) -> bytes:
    """Notation: a JSON file of this block, json.dumps(value, indent=2) + newline."""
    return (json.dumps(value, indent=2) + "\n").encode()


def read_json(path: Path) -> object:
    """Notation: a JSON file of this block read with json.loads."""
    return json.loads(path.read_bytes())


def remove(path: Path) -> None:
    """(6), (8): staging/ or staging/<k>/ removed, if present."""
    if path.exists():
        shutil.rmtree(path)


def record(subset: str, relpath: str, reason: str | None, **computed) -> dict:
    """Interface records.json: a record, every field null unless computed."""
    fields = {"subset": subset, "relpath": relpath, "reason": reason} | computed
    return dict.fromkeys(RECORD) | fields


def clip_record(
    subset: str, relpath: str, clip: Clip, classes: Mapping[int, str]
) -> dict:
    """Interface clip.json: id = <version>_<stem>_<first state line>; per state its
    class, null when its GUI flag is false."""
    return {
        "id": f"{episode_name(relpath)}_{clip.states[0]}",
        "subset": subset,
        "relpath": relpath,
        "version": version_of(relpath),
        "states": clip.states,
        "actions": clip.actions,
        "n": clip.n,
        "gui": clip.gui,
        "classes": [classes[t] if g else None for t, g in zip(clip.states, clip.gui)],
        "pitch": clip.pitch,
    }


def staging(out: Path, k: int) -> Path:
    """(2), (5): {out_dir}/staging/<k>, the staged clips of draw k (its video: <k>.mp4)."""
    return out / "staging" / str(k)


def line_starts(jsonl: bytes) -> list[int]:
    """Notation: the byte offset in J_e of each line l_t, then len(J_e) (twice when a
    final \\n ends the last line)."""
    ends = np.flatnonzero(np.frombuffer(jsonl, np.uint8) == ord("\n")) + 1
    return [0, *ends.tolist(), len(jsonl)]


def stage_clips(
    out: Path,
    k: int,
    subset: str,
    relpath: str,
    clips: Sequence[Clip],
    frames: Mapping[int, bytes],
    jsonl: bytes,
    classes: Mapping[int, str],
) -> list[str]:
    """(5): per clip, frames.yuv (its state frames in state order), lines.jsonl (the lines
    of its span as in J_e) and clip.json under staging/<k>/<id>.part/; the ids in order."""
    starts = line_starts(jsonl)
    ids = []
    for c in clips:
        row = clip_record(subset, relpath, c, classes)
        part = staging(out, k) / f"{row['id']}.part"
        part.mkdir(parents=True)
        (part / "frames.yuv").write_bytes(b"".join(frames[t] for t in c.states))
        (part / "lines.jsonl").write_bytes(
            jsonl[starts[c.states[0]] : starts[c.states[-1] + 1]]
        )
        (part / "clip.json").write_bytes(json_bytes(row))
        ids.append(row["id"])
    return ids


def write_draw(
    out: Path,
    k: int,
    records: list[dict],
    options: Mapping[str, bytes],
    state: dict,
    ids: Sequence[str],
) -> None:
    """(6): shard k by write_shard: records.json, state.json, options/<name>.json, the
    clips `ids` renamed from staging into clips/<id>/; then staging/<k>/ removed."""
    with write_shard(out / "shards", k) as staged:
        (staged / "records.json").write_bytes(json_bytes(records))
        (staged / "state.json").write_bytes(json_bytes(state))
        for name, data in options.items():
            (staged / "options").mkdir(exist_ok=True)
            (staged / "options" / f"{name}.json").write_bytes(data)
        for i in ids:
            (staged / "clips").mkdir(exist_ok=True)
            (staging(out, k) / f"{i}.part").rename(staged / "clips" / i)
    remove(staging(out, k))


class Resumed(NamedTuple):
    """(7): h; each pool without the relpaths of its subset's records in shards
    0..h-1; g; c; the reasons of those records (null as "null")."""

    h: int
    pools: Pools
    g: np.random.Generator
    clips: int
    reasons: Counter


def resume(shards: Path, seed: int, pools: Pools) -> Resumed:
    """(7): the h complete shards read one at a time, bar `resume`; g by `generator`, set
    to the state stored with shard h - 1, or Generator(PCG64(seed)) when h = 0."""
    h = complete_shards(shards)
    if h == 0:
        print("resume: nothing to do", file=sys.stderr)
        return Resumed(0, pools, generator(seed, None), 0, Counter())
    taken, reasons = set(), Counter()
    with tqdm(total=h, desc="resume", unit="shard") as bar:
        for k in range(h):
            for r in read_json(shard_path(shards, k) / "records.json"):
                taken.add((r["subset"], r["relpath"]))
                reasons[r["reason"] or "null"] += 1
            bar.update(1)
    state = read_json(shard_path(shards, h - 1) / "state.json")
    left = {
        s: (basedir, [r for r in pool if (s, r) not in taken])
        for s, (basedir, pool) in pools.items()
    }
    g = generator(seed, state["generator"])
    return Resumed(h, left, g, state["clips"], reasons)
