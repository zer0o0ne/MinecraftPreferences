"""(W) of specs/preference-pool.md: each draw's episodes run by (3) on `workers`
threads, the shards with the comparisons of (G) written in draw order; bar `collect`."""

from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mcpref import exact
from mcpref.clips.episode import Pipeline
from mcpref.clips.plan import Lookahead
from mcpref.clips.store import write_draw
from mcpref.pool.episode import Work, episode
from mcpref.pool.graph import comparisons
from mcpref.pool.plan import Resumed, Totals, draws
from mcpref.vpt.episodes import episode_name

if TYPE_CHECKING:
    from mcpref.pool.run import PreferencePoolConfig


@dataclass
class Draw:
    """(W): draw k: its session's key, its episodes' work, g's state at its end."""

    k: int
    key: dict
    jobs: list[Future]
    state: dict


def draw_files(
    cfg: "PreferencePoolConfig", d: Draw, work: list[Work], c: int
) -> tuple[dict, dict, list[str]]:
    """(W), (G): shard k's JSON files (records.json, state.json, comparisons.json), its
    options files by name and the ids of its clips (all n, none when n < 2)."""
    refs = [{"id": i, "episode": rec["relpath"]} for rec, ids, _ in work for i in ids]
    kept = [ref["id"] for ref in refs] if len(refs) >= 2 else []
    records = [
        rec | {"clips": ids if kept else []} if rec["reason"] is None else rec
        for rec, ids, _ in work
    ]
    pairs = []
    if kept:
        g_k = np.random.Generator(np.random.PCG64([cfg.seed, 5, d.k]))
        pairs = comparisons(len(kept), cfg.partners, exact("cycles", cfg.cycles), g_k)
    state = {"generator": d.state, "clips": c + len(kept), "session": d.key}
    files = {"records.json": records, "state.json": state}
    files["comparisons.json"] = [{"a": refs[a], "b": refs[b]} for a, b in pairs]
    options = {episode_name(rec["relpath"]): o for rec, _, o in work if o is not None}
    return files, options, kept


@dataclass
class Shards(Lookahead):
    """(W): the shards written in draw order: the totals."""

    cfg: "PreferencePoolConfig"
    out: Path
    totals: Totals

    @property
    def c(self) -> int:
        """(2): c, the clips in the shards (done: c >= N_p)."""
        return self.totals.clips

    def postfix(self) -> str:
        """Bar `collect`: `<n_e> episodes, <n_s> sessions`."""
        return self.totals.postfix()

    def write_next(self) -> None:
        """(W), (9'): the first pending draw, once its episodes are done (the exception
        of the first failing one in episode order raised here)."""
        d = self.pending.popleft()
        work = [job.result() for job in d.jobs]
        files, options, ids = draw_files(self.cfg, d, work, self.c)
        write_draw(self.out, d.k, files, options, ids)
        c, n_c = files["state.json"]["clips"], len(files["comparisons.json"])
        self.totals.add(files["records.json"], c, n_c)
        self.bar.update(len(ids))
        self.bar.set_postfix_str(self.postfix())


def run_plan(shards: Shards, r: Resumed, bases: dict[str, str], submit) -> None:
    """(2), (W): draws from h on, their episodes submitted in order, g's state taken at
    the end of each; shards written after each draw until c >= N_p."""
    for k, (session, pieces) in enumerate(draws(r, shards.cfg.episodes), start=r.h):
        s = session["subset"]
        jobs = [submit(k, j, s, bases[s], rel) for j, rel in enumerate(pieces)]
        key = {"subset": s, "start": session["pieces"][0]}
        shards.pending.append(Draw(k, key, jobs, r.g.bit_generator.state))
        shards.write()
        if shards.done():
            return


def write_shards(
    cfg: "PreferencePoolConfig", pipe: Pipeline, r: Resumed, bases: dict[str, str]
) -> Totals:
    """(2), (W), (9') continued after the complete shards by `Lookahead.run`; the totals
    over all shards returned."""
    shards = Shards(cfg.workers, cfg.n_clips, cfg, pipe.out, r.totals)
    plan, work = partial(run_plan, shards, r, bases), partial(episode, pipe, cfg)
    shards.run(r.h, any(r.pools.values()), plan, work)
    return shards.totals
