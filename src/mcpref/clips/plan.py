"""(1), (6), (9) of specs/perception-data.md: the draws, their candidates processed on
`workers` threads by (2)-(5), the shards written in draw order; bar `collect`."""

import sys
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import httpx
import numpy as np
from tqdm import tqdm

from mcpref import REPO
from mcpref.clips.config import load_config as load_clips_config
from mcpref.clips.episode import Candidate, Classify, Pipeline, process, serialised
from mcpref.clips.store import Resumed, rejected, staging, write_draw
from mcpref.gui.actions import load_config as load_gui_actions_config
from mcpref.gui.screens import load_screens
from mcpref.recognition.debug_overlay import load_config as load_overlay_config
from mcpref.recognition.debug_overlay import overlay_font, overlay_masks
from mcpref.vpt.actions import load_config as load_actions_config
from mcpref.vpt.downloads import fetch
from mcpref.vpt.episodes import Examined, Fetch, draw, episode_name, examine, options_ok

if TYPE_CHECKING:
    from mcpref.clips.collect import PerceptionDataConfig

STORED = ("malformed", "options", None)


@dataclass
class Draw:
    """(6): draw k: the records of its popped episodes but its candidate, the options
    files to store by name, g's state at its end, its candidate's processing if any."""

    k: int
    records: list[dict] = field(default_factory=list)
    options: dict[str, bytes] = field(default_factory=dict)
    state: dict | None = None
    jobs: list[Future] = field(default_factory=list)


@dataclass
class Lookahead:
    """(6): the draws not yet written in draw order, each with its submitted `jobs`; a
    subclass gives c, `postfix` (of bar `collect`) and `write_next` (the first draw)."""

    workers: int
    n_clips: int
    pending: deque = field(default_factory=deque, init=False)
    bar: tqdm = field(init=False)

    def done(self) -> bool:
        """(1): c >= N, N clips in the shards."""
        return self.c >= self.n_clips

    def write(self) -> None:
        """(6): the pending draws written in order while their jobs are done, then until
        fewer than `workers` jobs are not yet written; until done."""
        pending = self.pending
        while pending and not self.done() and all(j.done() for j in pending[0].jobs):
            self.write_next()
        while not self.done() and sum(len(d.jobs) for d in pending) >= self.workers:
            self.write_next()

    def run(self, h: int, active: bool, plan: Callable, work: Callable) -> None:
        """(6): `plan` submitting `work` to the threads, the rest written until done;
        bar `collect`, or its line when (1) has ended or no pool is active."""
        if self.done() or not active:
            line = "already done" if h else "nothing to do"
            print(f"collect: {line}", file=sys.stderr)
            return
        n, c = self.n_clips, self.c
        with tqdm(total=n, initial=c, desc="collect", unit="clip") as self.bar:
            self.bar.set_postfix_str(self.postfix())
            threads = ThreadPoolExecutor(self.workers)
            try:
                plan(partial(threads.submit, work))
                while self.pending and not self.done():
                    self.write_next()
            finally:
                threads.shutdown(cancel_futures=True)


def checked(
    cfg: "PerceptionDataConfig", fetch_file: Fetch, basedir: str, relpath: str
) -> Examined:
    """(1): the checks `version` ... `malformed` by `examine` without required options,
    then a non-object options file `malformed`, then `options` (Python ==)."""
    ex = examine(replace(cfg, required_options={}), fetch_file, basedir, relpath)
    if ex.reason is not None:
        return ex
    if not isinstance(ex.options, dict):
        return ex._replace(reason="malformed")
    if not options_ok(ex.options, cfg.required_options):
        return ex._replace(reason="options")
    return ex


def processed(
    pipe: Pipeline, k: int, subset: str, relpath: str, stem: str, ex: Examined
) -> tuple[dict, list[str]]:
    """(2)-(5) of the candidate of draw k: its video at staging/<k>.mp4, fetched on a
    client of its own, Generator(PCG64([seed, 1, k])), m = per_episode."""
    cfg = pipe.cfg
    video = staging(pipe.out, k).with_suffix(".mp4")
    g = np.random.Generator(np.random.PCG64([cfg.seed, 1, k]))
    cand = Candidate(k, subset, relpath, stem, ex, video, g, cfg.per_episode)
    with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}) as client:
        return process(pipe, cand, client)


def add(d: Draw, subset: str, basedir: str, relpath: str, ex: Examined, submit) -> None:
    """(1), (6): a popped episode: its options file kept for storing; the candidate
    submitted for (2)-(5), else its record."""
    if ex.reason in STORED:
        d.options[episode_name(relpath)] = ex.options_file
    if ex.reason is None:
        stem = basedir + relpath.removesuffix(".mp4")
        d.jobs = [submit(d.k, subset, relpath, stem, ex)]
    else:
        d.records.append(rejected(subset, relpath, ex))


@dataclass
class Shards(Lookahead):
    """(6): the shards in draw order: c, the reasons of their records (E in all)."""

    pipe: Pipeline
    c: int
    reasons: Counter

    def postfix(self) -> str:
        """Bar `collect`: `<E> episodes`."""
        return f"{self.reasons.total()} episodes"

    def write_next(self) -> None:
        """(6), (9): the first pending draw, once its candidate is processed (a worker's
        exception raised here), with the candidate's first N - c clips."""
        d, ids = self.pending.popleft(), []
        if d.jobs:
            rec, staged = d.jobs[0].result()
            ids = staged[: self.n_clips - self.c]
            d.records.append(rec | {"clips": ids if rec["reason"] is None else None})
        state = {"generator": d.state, "clips": self.c + len(ids)}
        files = {"records.json": d.records, "state.json": state}
        write_draw(self.pipe.out, d.k, files, d.options, ids)
        self.c += len(ids)
        self.reasons.update(r["reason"] or "null" for r in d.records)
        self.bar.update(len(ids))
        self.bar.set_postfix_str(self.postfix())


def run_plan(shards: Shards, r: Resumed, checks, submit) -> None:
    """(1), (6): the draws from h on, each ended by its candidate or an empty pool; g's
    state then taken; shards written after each popped episode until N clips."""
    d = Draw(r.h)
    for s, relpath, ex in draw(r.g, r.pools, checks):
        add(d, s, r.pools[s][0], relpath, ex, submit)
        if ex.reason is None or not r.pools[s][1]:
            d.state = r.g.bit_generator.state
            shards.pending.append(d)
            d = Draw(d.k + 1)
        shards.write()
        if shards.done():
            return


def pipeline(cfg: "PerceptionDataConfig", classify: Classify) -> Pipeline:
    """(2)-(5) on the workers: the configs named by cfg, the catalogue, P and Q of the
    overlay, gamma with its calls serialised."""
    clips = load_clips_config(REPO / cfg.clips)
    overlay = load_overlay_config(REPO / cfg.overlay)
    masks = overlay_masks(overlay_font(overlay), overlay.prefix, cfg.width, cfg.height)
    actions = load_actions_config(clips.actions)
    screens = load_screens(REPO / cfg.screens)
    double_click_ms = load_gui_actions_config(REPO / cfg.gui_actions).double_click_ms
    out, classify = REPO / cfg.out_dir, serialised(classify)
    return Pipeline(
        cfg, out, clips, actions, screens, double_click_ms, overlay, masks, classify
    )


def write_shards(
    pipe: Pipeline, client: httpx.Client, r: Resumed
) -> tuple[int, Counter]:
    """(1), (6), (9) continued after the complete shards by `Lookahead.run`; returns c
    and the reasons of all records."""
    cfg = pipe.cfg
    checks = partial(checked, cfg, partial(fetch, client, cfg))
    shards = Shards(cfg.workers, cfg.n_clips, pipe, r.clips, r.reasons)
    plan, work = partial(run_plan, shards, r, checks), partial(processed, pipe)
    shards.run(r.h, any(pool for _, pool in r.pools.values()), plan, work)
    return shards.c, shards.reasons
