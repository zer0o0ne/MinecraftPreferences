"""(1), (6), (9) of specs/perception-data.md: the draws of the plan, their candidates
processed on `workers` threads with the configs of (2)-(5), the shards written in draw
order; bar `collect`."""

import sys
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import httpx
from tqdm import tqdm

from mcpref import REPO
from mcpref.clips.config import load_config as load_clips_config
from mcpref.clips.episode import Candidate, Classify, Pipeline, process, serialised
from mcpref.clips.store import Resumed, record, write_draw
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
    files to store by name, g's state at its end, its candidate's processing."""

    k: int
    records: list[dict] = field(default_factory=list)
    options: dict[str, bytes] = field(default_factory=dict)
    state: dict | None = None
    candidate: Future | None = None


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


def add(d: Draw, subset: str, basedir: str, relpath: str, ex: Examined, submit) -> None:
    """(1), (6): a popped episode: its options file kept for storing; the candidate
    submitted for (2)-(5), else its record (options parsed for the reason `options`)."""
    if ex.reason in STORED:
        d.options[episode_name(relpath)] = ex.options_file
    if ex.reason is None:
        stem = basedir + relpath.removesuffix(".mp4")
        d.candidate = submit(Candidate(d.k, subset, relpath, stem, ex))
    else:
        options = ex.options if ex.reason == "options" else None
        d.records.append(record(subset, relpath, ex.reason, options=options))


@dataclass
class Shards:
    """(6): the shards written in draw order: c, the reasons of their records (E in all),
    the bar `collect`, the draws not yet written."""

    pipe: Pipeline
    clips: int
    reasons: Counter
    bar: tqdm
    pending: deque = field(default_factory=deque)

    def done(self) -> bool:
        """(1): N clips in the shards."""
        return self.clips >= self.pipe.cfg.n_clips

    def ready(self) -> bool:
        """(6): the candidate of the first pending draw, if any, is processed."""
        c = self.pending[0].candidate
        return c is None or c.done()

    def candidates(self) -> int:
        """(6): the candidates of the draws not yet written."""
        return sum(d.candidate is not None for d in self.pending)

    def write_next(self) -> None:
        """(6), (9): the first pending draw, once its candidate is processed (a worker's
        exception raised here), with the candidate's first N - c clips."""
        d, ids = self.pending.popleft(), []
        if d.candidate is not None:
            rec, staged = d.candidate.result()
            ids = staged[: self.pipe.cfg.n_clips - self.clips]
            d.records.append(rec | {"clips": ids if rec["reason"] is None else None})
        state = {"generator": d.state, "clips": self.clips + len(ids)}
        write_draw(self.pipe.out, d.k, d.records, d.options, state, ids)
        self.clips += len(ids)
        self.reasons.update(r["reason"] or "null" for r in d.records)
        self.bar.update(len(ids))
        self.bar.set_postfix_str(f"{self.reasons.total()} episodes")

    def write(self) -> None:
        """(6): the pending draws written in order while ready; then, before the next draw
        starts, until fewer than `workers` candidates are not yet written; until N."""
        while self.pending and not self.done() and self.ready():
            self.write_next()
        while not self.done() and self.candidates() >= self.pipe.cfg.workers:
            self.write_next()


def run_plan(shards: Shards, r: Resumed, checks, submit) -> None:
    """(1), (6): draws from h on, each ended by its candidate or an empty pool, g's state
    taken before the next draw starts; shards written after each popped episode, until N
    clips or the pools are empty; then the remaining draws written."""
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
    while shards.pending and not shards.done():
        shards.write_next()


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
    """(1), (6), (9) continued after the complete shards; returns c and the reasons of
    all records; bar `collect`, or its line when (1) has ended or nothing is active."""
    cfg, c = pipe.cfg, r.clips
    if c >= cfg.n_clips or not any(pool for _, pool in r.pools.values()):
        print(f"collect: {'already done' if r.h else 'nothing to do'}", file=sys.stderr)
        return c, r.reasons
    checks = partial(checked, cfg, partial(fetch, client, cfg))
    with tqdm(total=cfg.n_clips, initial=c, desc="collect", unit="clip") as bar:
        bar.set_postfix_str(f"{r.reasons.total()} episodes")
        shards = Shards(pipe, c, r.reasons, bar)
        threads = ThreadPoolExecutor(cfg.workers)
        try:
            run_plan(shards, r, checks, partial(threads.submit, process, pipe))
        finally:
            threads.shutdown(cancel_futures=True)
    return shards.clips, shards.reasons
