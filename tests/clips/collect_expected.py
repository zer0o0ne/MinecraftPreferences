"""Expected outputs of `collect` (specs/perception-data.md) for a world of collect_world.py:
(1) the draws and the reasons, (2)-(5) per candidate with collect_candidate.py, (6) one
shard per draw with its records, options files, clips and state; transcribed, nothing from
src/."""

import collections
import json
from dataclasses import dataclass

import numpy as np

from .collect_candidate import clip_files, process
from .collect_world import Ep

CHECKS = ("version", "missing", "empty", "malformed", "options")
NULLABLE = ("options", "starts_session", "overlay_start", "valid_starts", "clips")
JSON_FILES = ("records.json", "state.json", "clip.json")  # written by json.dumps


@dataclass
class Shard:
    """Shard k: its records, the generator state after draw k, c after it, and its files
    (path under the shard -> bytes, or the value of a file of JSON_FILES)."""

    records: list
    state: dict
    clips: int
    files: dict


@dataclass
class Expected:
    shards: list
    rows: list  # clip.json of every clip written, in shard order
    calls: dict  # stem of each candidate written -> the frames of its classifier calls
    draw: dict  # relpath of each record -> its draw k

    @property
    def records(self) -> list:
        return [r for s in self.shards for r in s.records]

    def tree(self, upto: int | None = None) -> dict:
        """Every file of shards 0..upto-1 (all if None): path under out_dir -> content."""
        return {f"shards/{k:06d}/{p}": x for k, s in enumerate(self.shards[:upto])
                for p, x in s.files.items()}  # fmt: skip

    def counts(self) -> dict:
        """run.json "episodes": reason or "null" -> count over every record."""
        return dict(collections.Counter(r["reason"] or "null" for r in self.records))

    def first(self, reason, clips: int = 1) -> int:
        """The first draw whose candidate has `reason` (and >= `clips` clips if valid)."""
        return next(k for k, s in enumerate(self.shards)
                    if s.records[-1]["reason"] == reason
                    and (reason or len(s.records[-1]["clips"]) >= clips))  # fmt: skip


def check(e: Ep, f: dict) -> str | None:
    """The checks of (1), the first failing one."""
    version = tuple(map(int, e.version.split(".")))
    if version < tuple(map(int, f["min_version"].split("."))):
        return "version"
    files = e.files()
    opts, jl = files["-options.json"], files[".jsonl"]
    if opts is None or jl is None:
        return "missing"
    if not opts or not jl:
        return "empty"
    try:
        parsed = json.loads(opts)
        [json.loads(x) for x in jl.decode().split("\n")[:-1]]
    except ValueError:
        return "malformed"
    if not isinstance(parsed, dict):
        return "malformed"
    if any(k not in parsed or parsed[k] != v for k, v in f["required_options"].items()):
        return "options"
    return None


def pop(out: Expected, files: dict, s: str, e: Ep, f: dict, k: int, c: int, broken):
    """One popped episode of draw k: its record and files; whether it is the candidate."""
    reason = check(e, f)
    rec = {"subset": s, "relpath": e.relpath, "reason": reason} | dict.fromkeys(
        NULLABLE
    )
    if reason == "options":
        rec["options"] = json.loads(e.files()["-options.json"])
    if reason is None:
        fields, rows, calls = process(e, f, k, s, broken)
        rows = rows[: f["n_clips"] - c]
        rec |= fields | ({"clips": [x["id"] for x in rows]} if rows else {})
        out.rows += rows
        out.calls[e.stem] = calls
        files |= {p: x for row in rows for p, x in clip_files(e, row).items()}
    if rec["reason"] not in CHECKS[:3]:
        files[f"options/{e.version}_{e.stem}.json"] = e.files()["-options.json"]
    out.draw[e.relpath] = k
    return rec


def expected(world, f: dict, broken=frozenset()) -> Expected:
    """(1), (6): draws until N clips are in the shards or no subset is active; `broken`:
    stems whose decode fails on every attempt of pass 1."""
    rng = np.random.Generator(np.random.PCG64(f["seed"]))
    pools = {s: list(dict.fromkeys(e.relpath for e in es))
             for s, es in world.episodes.items()}  # fmt: skip
    out, c = Expected([], [], {}, {}), 0
    while c < f["n_clips"] and any(pools.values()):
        k, records, files = len(out.shards), [], {}
        s = [s for s in pools if pools[s]][rng.integers(sum(map(bool, pools.values())))]
        while pools[s]:
            e = world.eps[s, pools[s].pop(rng.integers(len(pools[s])))]
            records.append(pop(out, files, s, e, f, k, c, broken))
            if records[-1]["reason"] not in CHECKS:
                break
        c += len(records[-1]["clips"] or [])
        state = rng.bit_generator.state
        files |= {"records.json": records,
                  "state.json": {"generator": state, "clips": c}}  # fmt: skip
        out.shards.append(Shard(records, state, c, files))
    return out
