"""(1), (6)-(8) of specs/perception-data.md: config, plan, commits, rerun, failures."""

import itertools
import math
import sys
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import httpx
import numpy as np
import yaml
from tqdm import tqdm

from mcpref import REPO, config_mapping, number, repo_path
from mcpref.clips.config import load_config as load_clips_config
from mcpref.clips.episode import Classify, Drawn, Pipeline, process, serialised
from mcpref.clips.store import (
    Episode,
    commit,
    committed_clips,
    episode_record,
    remove,
    replay,
    replayed,
    run_header,
)
from mcpref.gui.actions import load_config as load_gui_actions_config
from mcpref.gui.screens import load_screens
from mcpref.recognition.debug_overlay import load_config as load_overlay_config
from mcpref.recognition.debug_overlay import overlay_font, overlay_masks
from mcpref.vpt.actions import load_config as load_actions_config
from mcpref.vpt.config import check_rerun, write_run
from mcpref.vpt.downloads import download_indices, fetch
from mcpref.vpt.episodes import draw, examine, index

ANY, POSITIVE = (-math.inf, math.inf), (1, math.inf)
INTEGERS = {"width": ANY, "height": ANY, "fetch_streams": (1, 20), "http_retries": ANY}
INTEGERS |= {"states": (2, math.inf), "per_episode": POSITIVE, "n_clips": POSITIVE}
INTEGERS |= {"seed": (0, 2**32 - 1), "workers": POSITIVE, "classifier_batch": POSITIVE}
DURATIONS = ("http_timeout", "retry_delay", "ffmpeg_timeout")
CONFIGS = ("clips", "screens", "gui_actions", "overlay")
PATHS = (*CONFIGS, "out_dir")


@dataclass(frozen=True)
class PerceptionDataConfig:
    """Notation: configs/perception-data.yaml as written; states = L, per_episode = m,
    n_clips = N; paths relative to the repository root, resolved where used."""

    index_urls: dict[str, str]
    x_ms_version: str
    min_version: str
    required_options: dict[str, object]
    width: int
    height: int
    fetch_streams: int
    http_timeout: float
    http_retries: int
    retry_delay: float
    ffmpeg: str
    ffmpeg_timeout: float
    clips: str
    screens: str
    gui_actions: str
    overlay: str
    states: int
    per_episode: int
    n_clips: int
    seed: int
    workers: int
    classifier_batch: int
    out_dir: str


def integer(name: str, value: object, low: float, high: float) -> int:
    """Notation: a YAML integer, not a boolean, in [low, high]."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} {value!r} is not an integer in [{low}, {high}]")
    return value


def load_config(path: Path) -> PerceptionDataConfig:
    """Interface: the YAML file, validated, into PerceptionDataConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PerceptionDataConfig)
    for k, bounds in INTEGERS.items():
        integer(k, raw[k], *bounds)
    for k in DURATIONS:
        number(k, raw[k])
    for k in PATHS:
        repo_path(k, raw[k])
    return PerceptionDataConfig(**raw)


def pipeline(cfg: PerceptionDataConfig, classify: Classify) -> Pipeline:
    """(2)-(5): the configs named by cfg, the catalogue, P and Q of the overlay."""
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


def hashed_files(pipe: Pipeline) -> dict[str, Path]:
    """(7): the files whose SHA-256 run.json holds."""
    named = {k: REPO / getattr(pipe.cfg, k) for k in CONFIGS}
    overlay_screens = {"overlay_screens": pipe.overlay.screens}
    return {"actions": pipe.clips.actions} | named | overlay_screens


def drawn(
    cfg: PerceptionDataConfig, client: httpx.Client, indices: dict, records: list[dict]
) -> Iterator[Drawn]:
    """(1), (7): the draws of Generator(PCG64(seed)) with their positions r, after the
    committed episodes, whose draws are replayed."""
    fresh = partial(examine, cfg, partial(fetch, client, cfg))
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    draws = enumerate(draw(rng, indices, replayed(records, fresh)))
    for r, (subset, relpath, ex) in itertools.islice(draws, len(records), None):
        yield Drawn(r, subset, indices[subset][0] + relpath, relpath, ex)


def schedule(pipe: Pipeline, threads: ThreadPoolExecutor, d: Drawn) -> Future:
    """(1), (6): a valid episode submitted for (2)-(5); else a completed future of its
    record (options for the reason `options`) with its options file when read."""
    reason, options_file, options, _, _ = d.examined
    if reason is None:
        return threads.submit(process, pipe, d)
    record = episode_record(d.subset, d.relpath, reason, options=options)
    future = Future()
    future.set_result(Episode(d.r, record, options_file, []))
    return future


def in_order(
    pipe: Pipeline, draws: Iterator[Drawn], threads: ThreadPoolExecutor
) -> Iterator[Episode]:
    """(6): the drawn episodes in draw order, with up to `workers` valid ones processed
    ahead of the one to commit."""
    pending = deque()
    for d in draws:
        valid = d.examined.reason is None
        while valid and sum(v for _, v in pending) >= pipe.cfg.workers:
            yield pending.popleft()[0].result()
        pending.append((schedule(pipe, threads, d), valid))
        while pending and pending[0][0].done():
            yield pending.popleft()[0].result()
    while pending:
        yield pending.popleft()[0].result()


def commit_all(pipe: Pipeline, draws: Iterator[Drawn], records: list[dict]) -> None:
    """(6), (8): episodes committed in draw order until N clips, appended to `records`;
    bar `collect` enclosing the pool; nothing left (N committed, no draw left): a line."""
    cfg, c = pipe.cfg, committed_clips(records)
    if c >= cfg.n_clips or (first := next(draws, None)) is None:
        print("collect: already done", file=sys.stderr)
        return
    with tqdm(total=cfg.n_clips, initial=c, desc="collect", unit="clip") as bar:
        bar.set_postfix_str(f"{len(records)} episodes")
        threads = ThreadPoolExecutor(cfg.workers)
        try:
            for episode in in_order(pipe, itertools.chain([first], draws), threads):
                records.append(commit(pipe.out, episode, cfg.n_clips - c))
                k = committed_clips(records[-1:])
                c += k
                bar.update(k)
                bar.set_postfix_str(f"{len(records)} episodes")
                if c >= cfg.n_clips:
                    break
        finally:
            threads.shutdown(cancel_futures=True)


def collect(cfg: PerceptionDataConfig, classify: Classify, classifier_id: str) -> None:
    """Interface: (1), (6)-(8): run.json, the plan replayed and continued, episodes
    committed until N clips; the counts in run.json at the end."""
    pipe = pipeline(cfg, classify)
    out = pipe.out
    with httpx.Client(headers={"x-ms-version": cfg.x_ms_version}) as client:
        responses = dict(download_indices(client, cfg))
        etags = {s: response.headers.get("ETag") for s, response in responses.items()}
        header = run_header(cfg, hashed_files(pipe), etags, classifier_id)
        check_rerun(out / "run.json", header)
        out.mkdir(parents=True, exist_ok=True)
        write_run(out, header)
        records = replay(out)
        indices = {s: index(response.content) for s, response in responses.items()}
        commit_all(pipe, drawn(cfg, client, indices, records), records)
    remove(out / "staging")
    reasons = Counter(record["reason"] for record in records)
    write_run(out, header | {"episodes": reasons, "clips": committed_clips(records)})
