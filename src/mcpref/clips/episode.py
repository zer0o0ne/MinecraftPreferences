"""(2)-(5) of specs/perception-data.md: the candidate of a draw, its video fetched and
decoded, its frames classified, its clips staged."""

import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import numpy as np

from mcpref.clips.config import ClipsConfig
from mcpref.clips.evidence import evidence, kept
from mcpref.clips.sample import Clip, gui_flag, sample
from mcpref.clips.store import record, stage_clips
from mcpref.clips.validity import cache_intervals, valid_starts
from mcpref.gui.actions import excluded_states, gui_actions, needed_frames, spans
from mcpref.gui.screens import ScreenClass
from mcpref.http.retry import retried
from mcpref.recognition.debug_overlay import OverlayConfig, luma, overlay_shown
from mcpref.vpt.actions import ActionsConfig, starts_session
from mcpref.vpt.downloads import fetch_to_path
from mcpref.vpt.episodes import Examined
from mcpref.vpt.ffmpeg import decode_episode
from mcpref.vpt.yuv import yuv420p_to_rgb

if TYPE_CHECKING:
    from mcpref.clips.collect import PerceptionDataConfig

Classify = Callable[[list[np.ndarray]], list[str]]


class Candidate(NamedTuple):
    """(1): the candidate of draw k, its URL stem, its examination; (2), (5): the path of
    its video, its generator and m (specs/preference-pool.md (3): its own)."""

    k: int
    subset: str
    relpath: str
    stem: str
    examined: Examined
    video: Path
    g: np.random.Generator
    m: int


def serialised(classify: Classify) -> Classify:
    """(3): calls of gamma serialised across workers."""
    lock = threading.Lock()

    def call(frames: list[np.ndarray]) -> list[str]:
        with lock:
            return classify(frames)

    classify_yuv = getattr(classify, "classify_yuv", None)
    if classify_yuv is not None:

        def call_yuv(frames: list[bytes]) -> list[str]:
            with lock:
                return classify_yuv(frames)

        call.classify_yuv = call_yuv
    return call


@dataclass(frozen=True)
class Pipeline:
    """(2)-(5): the run's configs, catalogue, overlay masks, serialised gamma and shared
    bounded range-fetch executor; the runner that creates it closes it on every exit."""

    cfg: "PerceptionDataConfig"
    out: Path
    clips: ClipsConfig
    actions: ActionsConfig
    screens: dict[str, ScreenClass]
    double_click_ms: int
    overlay: OverlayConfig
    masks: tuple[np.ndarray, np.ndarray]
    classify: Classify
    ranges: ThreadPoolExecutor

    def close(self) -> None:
        """Shut down this pipeline's range-fetch executor."""
        self.ranges.shutdown(wait=True, cancel_futures=True)


def fetch_video(
    client: httpx.Client,
    cfg: "PerceptionDataConfig",
    url: str,
    video: Path,
    threads: ThreadPoolExecutor,
) -> bool:
    """(2): the .mp4 streamed by ranged fetch (f) into `video`; False for a 404."""
    return fetch_to_path(client, cfg, url, video, threads)


def pass_1(
    cfg: "PerceptionDataConfig", video: Path, lines: Sequence[Mapping]
) -> dict[int, bytes] | None:
    """(2): frame 0 and every frame t <= T - 1 with isGuiOpen of l_t, by a decode of
    T + 1 frames retried after retry_delay up to http_retries times; None: `video`; a
    timed-out decode raises TimeoutError."""
    keep = {0} | {t for t, line in enumerate(lines) if line["isGuiOpen"]}
    run = partial(decode_episode, cfg, video, len(lines) + 1, keep, cfg.ffmpeg_timeout)
    return retried(run, lambda f: f is None, cfg.http_retries, cfg.retry_delay)


def pass_2(
    cfg: "PerceptionDataConfig", video: Path, n: int, clips: Sequence[Clip]
) -> dict[int, bytes]:
    """(5): the frames of the taken clips' states by one run of the same decode, not
    retried; a failure raises RuntimeError, a timed-out decode TimeoutError."""
    keep = {t for c in clips for t in c.states}
    frames = decode_episode(cfg, video, n, keep, cfg.ffmpeg_timeout, select=True)
    if frames is None:
        raise RuntimeError(f"pass 2 decode of {video} failed")
    return frames


def classify_frames(
    pipe: Pipeline, frames: Mapping[int, bytes], ts: Sequence[int], classes: dict
) -> None:
    """(3), (5): the frames ts classified in their order, in consecutive chunks of
    classifier_batch frames, the last one shorter; their classes kept in `classes`."""
    cfg, b = pipe.cfg, pipe.cfg.classifier_batch
    classify_yuv = getattr(pipe.classify, "classify_yuv", None)
    for j in range(0, len(ts), b):
        chunk = ts[j : j + b]
        raw = [frames[t] for t in chunk]
        if classify_yuv is None:
            rgb = [yuv420p_to_rgb(x, cfg.width, cfg.height) for x in raw]
            labels = pipe.classify(rgb)
        else:
            labels = classify_yuv(raw)
        classes.update(zip(chunk, labels, strict=True))


def accept(
    pipe: Pipeline,
    lines: Sequence[Mapping],
    frames: Mapping[int, bytes],
    classes: dict,
    cut: frozenset[int],
    states: list[int],
    *,
    excluded: set[int] | None = None,
) -> bool:
    """(5) `accept`: the not yet classified states with the GUI flag true classified (in
    state order), and no state in `excluded_states`."""
    todo = [t for t in states if gui_flag(lines, t) and t not in classes]
    classify_frames(pipe, frames, todo, classes)
    if excluded is None:
        return excluded_states(lines, classes, cut).isdisjoint(states)
    new_classes = {t: classes[t] for t in todo}
    excluded.update(excluded_states(lines, new_classes, cut))
    return excluded.isdisjoint(states)


def start_modes(pipe: Pipeline, session: bool, frame0: bytes) -> tuple[bool, bool, int]:
    """(4): (False, False, 0) if the episode starts a session, else (overlay_shown on
    luma of frame 0, False, 0)."""
    if session:
        return False, False, 0
    y = luma(frame0, pipe.cfg.width, pipe.cfg.height)
    return overlay_shown(y, pipe.masks, pipe.overlay), False, 0


def take_clips(
    pipe: Pipeline,
    cand: Candidate,
    frames: Mapping[int, bytes],
    session: bool,
) -> tuple[dict, list[Clip], dict]:
    """(3)-(5): classes, GUI actions, start, evidence, valid starts, at most m clips by
    the candidate's generator; returns computed record fields, clips, classes."""
    cfg, classes, lines = pipe.cfg, {}, cand.examined.lines
    gui_spans = spans(lines)
    needed = needed_frames(lines, gui_spans=gui_spans)
    classify_frames(pipe, frames, needed, classes)
    gui = gui_actions(
        lines,
        classes,
        pipe.screens,
        pipe.double_click_ms,
        gui_spans=gui_spans,
        frames=needed,
    )
    start = start_modes(pipe, session, frames[0])
    ev = evidence(lines, gui, start, pipe.actions)
    kappa, delta = kept(ev), pipe.clips.delta
    cache = cache_intervals(ev, kappa, delta)
    starts = valid_starts(ev, kappa, delta, cfg.states, cache=cache)
    excluded = set(excluded_states(lines, classes, gui.cut))
    accepted = partial(accept, pipe, lines, frames, classes, gui.cut, excluded=excluded)
    clips = sample(
        ev, kappa, starts, delta, cfg.states, cand.m, cand.g, accepted, cache=cache
    )
    return {"overlay_start": start[0], "valid_starts": len(starts)}, clips, classes


def from_video(
    pipe: Pipeline, cand: Candidate, session: bool, client: httpx.Client
) -> tuple[str | None, dict, list[str]]:
    """(2)-(5): the reason (`video`: a 404 or a failed decode; `no_clip`), the computed
    record fields and the ids of the staged clips in the order taken."""
    k, subset, relpath, stem, (_, _, _, jsonl, lines), video, _, _ = cand
    ok = fetch_video(client, pipe.cfg, stem + ".mp4", video, pipe.ranges)
    frames = pass_1(pipe.cfg, video, lines) if ok else None
    if frames is None:
        return "video", {}, []
    fields, clips, classes = take_clips(pipe, cand, frames, session)
    if not clips:
        return "no_clip", fields, []
    del frames
    frames = pass_2(pipe.cfg, video, len(lines) + 1, clips)
    ids = stage_clips(pipe.out, k, subset, relpath, clips, frames, jsonl, classes)
    return None, fields, ids


def process(
    pipe: Pipeline, cand: Candidate, client: httpx.Client
) -> tuple[dict, list[str]]:
    """(2)-(5): the candidate's record (`clips` set when its shard is written) and the
    ids of its staged clips; its video fetched on `client`, deleted in every case."""
    session = starts_session(cand.examined.lines)
    try:
        reason, fields, ids = from_video(pipe, cand, session, client)
    finally:
        cand.video.unlink(missing_ok=True)
    computed = {"options": cand.examined.options, "starts_session": session} | fields
    return record(cand.subset, cand.relpath, reason, **computed), ids
