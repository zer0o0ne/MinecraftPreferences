"""(2)-(5) of specs/perception-data.md: one valid episode, its video to its clips."""

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import httpx
import numpy as np

from mcpref.clips.config import ClipsConfig
from mcpref.clips.evidence import evidence, kept
from mcpref.clips.sample import Clip, gui_flag, sample
from mcpref.clips.store import (
    Episode,
    clip_record,
    episode_record,
    stage_clips,
    staging,
)
from mcpref.clips.validity import valid_starts
from mcpref.gui.actions import excluded_states, gui_actions, needed_frames
from mcpref.gui.screens import ScreenClass
from mcpref.http.retry import retried
from mcpref.recognition.debug_overlay import OverlayConfig, luma, overlay_shown
from mcpref.vpt.actions import ActionsConfig, starts_session
from mcpref.vpt.downloads import fetch
from mcpref.vpt.episodes import Examined
from mcpref.vpt.ffmpeg import decode_episode
from mcpref.vpt.yuv import yuv420p_to_rgb

if TYPE_CHECKING:
    from mcpref.clips.collect import PerceptionDataConfig

Classify = Callable[[list[np.ndarray]], list[str]]


class Drawn(NamedTuple):
    """(1): the episode drawn at position r, the URL of its .mp4, its examination."""

    r: int
    subset: str
    url: str
    relpath: str
    examined: Examined


def serialised(classify: Classify) -> Classify:
    """(3): calls of gamma serialised across workers."""
    lock = threading.Lock()

    def call(frames: list[np.ndarray]) -> list[str]:
        with lock:
            return classify(frames)

    return call


@dataclass(frozen=True)
class Pipeline:
    """(2)-(5): the config with out_dir resolved, the configs it names, the catalogue,
    the overlay masks (P, Q) and gamma (calls serialised by `serialised`)."""

    cfg: "PerceptionDataConfig"
    out: Path
    clips: ClipsConfig
    actions: ActionsConfig
    screens: dict[str, ScreenClass]
    double_click_ms: int
    overlay: OverlayConfig
    masks: tuple[np.ndarray, np.ndarray]
    classify: Classify


def classify_frames(
    pipe: Pipeline, frames: Mapping[int, bytes], ts: Sequence[int], classes: dict
) -> None:
    """(3), (5): the frames ts classified in their order, in consecutive chunks of
    classifier_batch frames, the last one shorter; their classes kept in `classes`."""
    cfg, b = pipe.cfg, pipe.cfg.classifier_batch
    for k in range(0, len(ts), b):
        chunk = ts[k : k + b]
        rgb = [yuv420p_to_rgb(frames[t], cfg.width, cfg.height) for t in chunk]
        classes.update(zip(chunk, pipe.classify(rgb), strict=True))


def accept(
    pipe: Pipeline,
    lines: Sequence[Mapping],
    frames: Mapping[int, bytes],
    classes: dict,
    cut: frozenset[int],
    states: list[int],
) -> bool:
    """(5) `accept`: the not yet classified states with the GUI flag true classified (in
    state order), and no state in `excluded_states`."""
    todo = [t for t in states if gui_flag(lines, t) and t not in classes]
    classify_frames(pipe, frames, todo, classes)
    return excluded_states(lines, classes, cut).isdisjoint(states)


def start_modes(pipe: Pipeline, session: bool, frame0: bytes) -> tuple[bool, bool, int]:
    """(4): (False, False, 0) if the episode starts a session, else (overlay_shown on
    luma of frame 0, False, 0)."""
    if session:
        return False, False, 0
    y = luma(frame0, pipe.cfg.width, pipe.cfg.height)
    return overlay_shown(y, pipe.masks, pipe.overlay), False, 0


def pass_1(
    pipe: Pipeline, video: Path, lines: Sequence[Mapping]
) -> dict[int, bytes] | None:
    """(2): frame 0 and every frame t <= T - 1 with isGuiOpen of l_t, by a decode of
    T + 1 frames retried after retry_delay up to http_retries times; None: `video`."""
    cfg = pipe.cfg
    keep = {0} | {t for t, line in enumerate(lines) if line["isGuiOpen"]}
    run = partial(decode_episode, cfg, video, len(lines) + 1, keep, cfg.ffmpeg_timeout)
    return retried(run, lambda f: f is None, cfg.http_retries, cfg.retry_delay)


def pass_2(
    pipe: Pipeline, video: Path, n: int, clips: Sequence[Clip]
) -> dict[int, bytes]:
    """(5): the frames of the taken clips' states by the same decode; failure aborts."""
    keep = {t for c in clips for t in c.states}
    frames = decode_episode(pipe.cfg, video, n, keep, pipe.cfg.ffmpeg_timeout)
    if frames is None:
        raise RuntimeError(f"pass 2 decode of {video} failed")
    return frames


def take_clips(
    pipe: Pipeline,
    r: int,
    lines: Sequence[Mapping],
    frames: Mapping[int, bytes],
    session: bool,
) -> tuple[dict, list[Clip], dict]:
    """(3)-(5): classes, GUI actions, start, evidence, valid starts, clips sampled with
    Generator(PCG64([seed, 1, r])); returns computed record fields, clips, classes."""
    cfg, classes = pipe.cfg, {}
    classify_frames(pipe, frames, needed_frames(lines), classes)
    gui = gui_actions(lines, classes, pipe.screens, pipe.double_click_ms)
    start = start_modes(pipe, session, frames[0])
    ev = evidence(lines, gui, start, pipe.actions)
    kappa, delta = kept(ev), pipe.clips.delta
    starts = valid_starts(ev, kappa, delta, cfg.states)
    g = np.random.Generator(np.random.PCG64([cfg.seed, 1, r]))
    accepted = partial(accept, pipe, lines, frames, classes, gui.cut)
    clips = sample(ev, kappa, starts, delta, cfg.states, cfg.per_episode, g, accepted)
    return {"overlay_start": start[0], "valid_starts": len(starts)}, clips, classes


def fetch_video(pipe: Pipeline, url: str, video: Path) -> bool:
    """(2): the .mp4 by the ranged fetch (f), on a client of its own, into `video`;
    False for a 404."""
    with httpx.Client(headers={"x-ms-version": pipe.cfg.x_ms_version}) as client:
        data = fetch(client, pipe.cfg, url)
    if data is None:
        return False
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(data)
    return True


def from_video(
    pipe: Pipeline, drawn: Drawn, session: bool, video: Path
) -> tuple[str | None, dict, list[dict]]:
    """(2)-(5): the reason (`video`: a 404 or a failed decode), the computed record
    fields and the records of the staged clips."""
    r, subset, url, relpath, (_, _, _, jsonl, lines) = drawn
    frames = pass_1(pipe, video, lines) if fetch_video(pipe, url, video) else None
    if frames is None:
        return "video", {}, []
    fields, clips, classes = take_clips(pipe, r, lines, frames, session)
    if not clips:
        return "no_clip", fields, []
    frames = pass_2(pipe, video, len(lines) + 1, clips)
    stage_clips(pipe.out, r, relpath, clips, frames, jsonl)
    return None, fields, [clip_record(subset, relpath, c, classes) for c in clips]


def process(pipe: Pipeline, drawn: Drawn) -> Episode:
    """(2)-(5) for a valid episode: its record and staged clips; its video,
    staging/<r>.mp4, deleted in every case."""
    examined, video = drawn.examined, staging(pipe.out, drawn.r).with_suffix(".mp4")
    session = starts_session(examined.lines)
    try:
        reason, fields, clips = from_video(pipe, drawn, session, video)
    finally:
        video.unlink(missing_ok=True)
    computed = {"options": examined.options, "starts_session": session} | fields
    record = episode_record(drawn.subset, drawn.relpath, reason, **computed)
    return Episode(drawn.r, record, examined.options_file, clips)
