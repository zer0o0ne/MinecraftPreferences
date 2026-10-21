"""(2), (P) of specs/preference-pool.md: the pools of sessions, the continuation after
the complete shards, the draws."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mcpref.clips.sample import uniform
from mcpref.clips.store import read_json
from mcpref.recognition.stored import shard_bar
from mcpref.shards.store import complete_shards, shard_path
from mcpref.vpt.plan import generator

Sessions = dict[str, list[dict]]
"""(2): per subset in config order, its pool of sessions {subset, pieces}."""


def session_pools(subsets: list[str], d: Path) -> Sessions:
    """(P): the sessions of the head shards of d, read one at a time, in the order of
    sessions/, pooled per subset."""
    pools = {s: [] for s in subsets}
    for eta in range(complete_shards(d)):
        for session in read_json(shard_path(d, eta) / "sessions.json")["sessions"]:
            pools[session["subset"]].append(session)
    return pools


@dataclass
class Totals:
    """(P), (W), (8'): over the complete shards, c, n_s, n_c and the reasons of their
    records (null as "null"), n_e in all."""

    clips: int = 0
    sessions: int = 0
    comparisons: int = 0
    reasons: Counter = field(default_factory=Counter)

    def add(self, records: list[dict], c: int, n_c: int) -> None:
        """A shard's records and n_c comparisons counted; c after it."""
        self.clips = c
        self.sessions += 1
        self.comparisons += n_c
        self.reasons.update(r["reason"] or "null" for r in records)

    def postfix(self) -> str:
        """Bar `collect`: `<n_e> episodes, <n_s> sessions`."""
        return f"{self.reasons.total()} episodes, {self.sessions} sessions"

    def counts(self) -> dict:
        """(8'): the counts of run.json."""
        return {
            "episodes": dict(self.reasons),
            "sessions": self.sessions,
            "clips": self.clips,
            "comparisons": self.comparisons,
        }


class Resumed(NamedTuple):
    """(P): h; the pools without the sessions of the complete shards; g; the totals."""

    h: int
    pools: Sessions
    g: np.random.Generator
    totals: Totals


def resume(shards: Path, seed: int, pools: Sessions) -> Resumed:
    """(P): the h complete shards read one at a time, bar `resume`; g set to the state
    of shard h - 1, or Generator(PCG64([seed, 3])) when h = 0; c to its clips."""
    h, totals, keys = complete_shards(shards), Totals(), set()
    state = {"generator": None}
    with shard_bar("resume", h, 0, h, 0, "shard") as update:
        for k in range(h):
            path = shard_path(shards, k)
            n_c = len(read_json(path / "comparisons.json"))
            records = read_json(path / "records.json")
            state = read_json(path / "state.json")
            totals.add(records, state["clips"], n_c)
            keys.add((state["session"]["subset"], state["session"]["start"]))
            update()
    left = {
        s: [x for x in pool if (s, x["pieces"][0]) not in keys]
        for s, pool in pools.items()
    }
    return Resumed(h, left, generator([seed, 3], state["generator"]), totals)


def draws(r: Resumed, e: int) -> Iterator[tuple[dict, list[str]]]:
    """(2): while a subset is active, a session popped from an active subset's pool, its
    episodes: all its pieces if at most e, else those at sorted(permutation[:e])."""
    while active := [s for s, pool in r.pools.items() if pool]:
        pool = r.pools[active[uniform(r.g, len(active))]]
        session = pool.pop(uniform(r.g, len(pool)))
        pieces = session["pieces"]
        if len(pieces) > e:
            chosen = sorted(r.g.permutation(len(pieces))[:e].tolist())
            pieces = [pieces[i] for i in chosen]
        yield session, pieces
