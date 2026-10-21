"""Expected outputs of `collect` of specs/preference-pool.md for a world of pool_world.py,
transcribed, nothing from src/: (1) the heads from the served bytes, the entries and groups
of Notation, (S), the head shards of (H), the draws (2), each episode by (3) with the
transcription of perception-data (1)-(5) of tests/clips, (G) by graph_oracle.py, and the
shards of (W)."""

import collections
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath

import numpy as np
from clips.collect_candidate import clip_files, process
from clips.collect_expected import NULLABLE, check

from .graph_oracle import graph

STATUSES = ("ok", "missing", "empty", "malformed")


def head_of(data: bytes | None) -> dict:
    """(1) on the served bytes of a `.jsonl` (None: no file)."""
    if data is None or data == b"":
        return {"status": "missing" if data is None else "empty"}
    try:
        x = json.loads(data.split(b"\n", 1)[0])
    except ValueError:
        return {"status": "malformed"}
    ints = isinstance(x, dict) and all(type(x.get(k)) is int for k in ("serverTick", "milli"))  # fmt: skip
    if not (ints and isinstance(x.get("stats"), dict)):
        return {"status": "malformed"}
    return {"status": "ok", "stats": x["stats"], "serverTick": x["serverTick"], "milli": x["milli"]}  # fmt: skip


def version(relpath: str) -> tuple:
    return tuple(map(int, relpath.split("/")[1].split(".")))


def entries(world, p: dict) -> list[tuple[str, str]]:
    """(subset, relpath) of every entry: the initial pools in config order without the
    relpaths below min_version."""
    low = tuple(map(int, p["min_version"].split(".")))
    return [(s, r) for s, es in world.episodes.items()
            for r in dict.fromkeys(e.relpath for e in es) if version(r) >= low]  # fmt: skip


def groups(items: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Entries of the same (subset, alias, hex), names split from the right; in order of
    their first entry."""
    out = {}
    for s, r in items:
        alias, hexa, _, _ = PurePosixPath(r).stem.rsplit("-", 3)
        out.setdefault((s, alias, hexa), []).append((s, r))
    return list(out.values())


def joins(i: dict, j: dict, q: dict) -> bool:
    """(S): j joins the chain whose last piece is i; exact arithmetic."""
    d = j["serverTick"] - i["serverTick"]
    bound = q["tick_ms"] * d * (1 - Fraction(str(q["tick_slack"])))
    kept = all(k in q["resetting_stats"] or (k in j["stats"] and j["stats"][k] >= v)
               for k, v in i["stats"].items())  # fmt: skip
    return d > 0 and bound <= j["milli"] - i["milli"] and kept


def sessions_of(group: list[tuple[str, dict]], q: dict) -> list[list[str]]:
    """(S) on one group's (relpath, head) pairs: the sessions' relpaths."""
    ok = sorted((h["milli"], r, h) for r, h in group if h["status"] == "ok")
    chains, current = [], None
    for _, r, h in ok:
        if h["stats"] and current and joins(current[-1][1], h, q):
            current.append((r, h))
        else:
            current = [(r, h)]
            chains.append((not h["stats"], current))
    return [[r for r, _ in c] for start, c in chains if start]


def head_shards(gs: list, per_shard: int) -> list[list]:
    """Consecutive groups; a new head shard after the group reaching `per_shard` entries."""
    shards, current = [], []
    for g in gs:
        current.append(g)
        if sum(map(len, current)) >= per_shard:
            shards.append(current)
            current = []
    return shards + ([current] if current else [])


@dataclass
class Shard:
    """Shard k: files (path under the shard -> bytes, or the value of a JSON file), and
    the counts after it."""

    files: dict
    clips: int

    @property
    def records(self) -> list:
        return self.files["records.json"]

    @property
    def comparisons(self) -> list:
        return self.files["comparisons.json"]


@dataclass
class Expected:
    heads: dict  # relpath -> head
    sessions: list  # sessions.json of each head shard
    entries: list  # relpaths of each head shard, in its order
    shards: list

    def tree(self) -> dict:
        """Every file of sessions/ and shards/: path under out_dir -> content."""
        out = {
            f"sessions/{k:06d}/sessions.json": x for k, x in enumerate(self.sessions)
        }
        return out | {f"shards/{k:06d}/{p}": x for k, s in enumerate(self.shards)
                      for p, x in s.files.items()}  # fmt: skip

    def counts(self) -> dict:
        """The counts of run.json (8'), in order."""
        records = [r for s in self.shards for r in s.records]
        return {"episodes": dict(collections.Counter(r["reason"] or "null" for r in records)),
                "sessions": len(self.shards), "clips": self.shards[-1].clips if self.shards else 0,
                "comparisons": sum(len(s.comparisons) for s in self.shards)}  # fmt: skip


def episode(world, f: dict, q: dict, k: int, j: int, s: str, r: str, files: dict):
    """(3) of episode j of draw k: its record and its clips in the order taken."""
    e = world.eps[s, r]
    reason = check(e, f | {"min_version": "0.0"})
    rec = {"subset": s, "relpath": r, "reason": reason} | dict.fromkeys(NULLABLE)
    rows = []
    if reason == "options":
        rec["options"] = json.loads(e.files()["-options.json"])
    if reason is None:
        fields, rows, _ = process(e, f, k, s, frozenset(), key=[q["seed"], 4, k, j])
        rec |= fields
    if reason not in ("missing", "empty"):
        files[f"options/{e.version}_{e.stem}.json"] = e.files()["-options.json"]
    return rec, rows


def shard(world, f: dict, q: dict, k: int, s: str, pieces: list, chosen: list, c: int, state: dict) -> Shard:  # fmt: skip
    """(W) of draw k, c the clips before it."""
    files, records, clips = {}, [], []
    for j, r in enumerate(chosen):
        rec, rows = episode(world, f, q, k, j, s, r, files)
        records.append((rec, rows))
        clips += rows
    written = clips if len(clips) >= 2 else []
    for rec, rows in records:
        if rec["reason"] is None:
            rec["clips"] = [x["id"] for x in rows] if written else []
    for row in written:
        files |= clip_files(world.eps[s, row["relpath"]], row)
    comparisons = []
    if written:
        g = np.random.Generator(np.random.PCG64([q["seed"], 5, k]))
        pairs = graph(len(written), q["partners"], Fraction(str(q["cycles"])), g)
        ref = lambda x: {"id": written[x]["id"], "episode": written[x]["relpath"]}
        comparisons = [{"a": ref(a), "b": ref(b)} for a, b in pairs]
    c += len(written)
    files |= {"records.json": [rec for rec, _ in records], "comparisons.json": comparisons,
              "state.json": {"generator": state, "clips": c,
                             "session": {"subset": s, "start": pieces[0]}}}  # fmt: skip
    return Shard(files, c)


def draws(world, f: dict, q: dict, pools: dict) -> list[Shard]:
    """(2): draws until c >= N_p or no subset is active, each written whole."""
    g = np.random.Generator(np.random.PCG64([q["seed"], 3]))
    out, c = [], 0
    while c < q["n_clips"] and any(pools.values()):
        active = [s for s in pools if pools[s]]
        s = active[g.integers(len(active))]
        pieces = pools[s].pop(g.integers(len(pools[s])))
        chosen = pieces
        if len(pieces) > q["episodes"]:
            chosen = [pieces[i] for i in sorted(g.permutation(len(pieces))[: q["episodes"]])]  # fmt: skip
        state = g.bit_generator.state
        out.append(shard(world, f, q, len(out), s, pieces, chosen, c, state))
        c = out[-1].clips
    return out


def expected(world, p: dict, q: dict) -> Expected:
    """`p`: every value of the perception-data config and its sub-configs; `q`: the pool's."""
    heads = {
        r: head_of(world.eps[s, r].files()[".jsonl"]) for s, r in entries(world, p)
    }
    sessions, order, pools = [], [], {s: [] for s in world.episodes}
    for gs in head_shards(groups(entries(world, p)), q["entries_per_shard"]):
        found = [(s, x) for g in gs for s in [g[0][0]] for x in sessions_of([(r, heads[r]) for _, r in g], q)]  # fmt: skip
        counts = collections.Counter(heads[r]["status"] for g in gs for _, r in g)
        sessions.append({"sessions": [{"subset": s, "pieces": x} for s, x in found],
                         "entries": {k: counts[k] for k in STATUSES}})  # fmt: skip
        order.append([r for g in gs for _, r in g])
        for s, x in found:
            pools[s].append(x)
    f = p | {"per_episode": q["per_episode"]}
    return Expected(heads, sessions, order, draws(world, f, q, pools))
