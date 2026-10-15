"""(4), (c), (e) of specs/vpt-gui-frames.md: frames extracted and decoded by ffmpeg;
the whole-episode decode of (2) of specs/perception-data.md."""

import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Container
from functools import partial
from pathlib import Path
from typing import BinaryIO

from tqdm import tqdm

from mcpref.http.retry import retried
from mcpref.shards.store import publish
from mcpref.vpt.config import Access, GuiFramesConfig


def frame_time(i: int, fps: int) -> str:
    """(4): seek time t_i of frame i, "%d.%02d" % (i // fps, (100 // fps) * (i % fps))."""
    return f"{i // fps:d}.{(100 // fps) * (i % fps):02d}"


def ffmpeg_command(cfg: GuiFramesConfig, url: str, i: int, out: Path) -> list[str]:
    """Interface: ffmpeg writing frame i of `url` as raw yuv420p to `out`."""
    return (
        [cfg.ffmpeg, "-nostdin", "-loglevel", "error", "-y"]
        + ["-headers", f"x-ms-version: {cfg.x_ms_version}\r\n"]
        + ["-rw_timeout", str(round(cfg.ffmpeg_rw_timeout * 10**6))]
        + ["-ss", frame_time(i, cfg.fps), "-i", url, "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "yuv420p", str(out)]
    )


def frame_bytes(cfg: Access) -> int:
    """(c): bytes of one raw yuv420p frame, width x height x 3 / 2."""
    return cfg.width * cfg.height * 3 // 2


def show(stderr: BinaryIO) -> None:
    """Interface progress bars: ffmpeg's captured stderr, when non-empty, as one tqdm.write."""
    stderr.seek(0)
    if text := stderr.read():
        tqdm.write(text.decode("utf-8", errors="replace").rstrip("\n"), file=sys.stderr)


def ffmpeg_ok(cfg: GuiFramesConfig, url: str, i: int, part: Path) -> bool:
    """(c): one run within ffmpeg_timeout, exit code 0 and exactly one frame of bytes."""
    cmd = ffmpeg_command(cfg, url, i, part)
    with tempfile.TemporaryFile() as stderr:
        try:
            code = subprocess.run(
                cmd, stderr=stderr, timeout=cfg.ffmpeg_timeout, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            code = None
        show(stderr)
    return code == 0 and part.exists() and part.stat().st_size == frame_bytes(cfg)


def extract(cfg: GuiFramesConfig, url: str, i: int, path: Path) -> bool:
    """(c): ffmpeg into <path>.part, retried; published as `path` on success
    (specs/shards.md (3))."""
    part = path.with_name(path.name + ".part")
    run = partial(ffmpeg_ok, cfg, url, i, part)
    ok = retried(run, lambda ok: not ok, cfg.http_retries, cfg.retry_delay)
    if ok:
        publish(part, path)
    return ok


def decode(cfg: GuiFramesConfig, video: Path) -> dict[int, bytes]:
    """(e): the verify frames of a sequential decode of `video` to raw yuv420p; bar `decode`."""
    cmd = [cfg.ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(video)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    frames, size = {}, frame_bytes(cfg)
    with (
        tqdm(total=None, desc="decode", unit="frame") as bar,
        tempfile.TemporaryFile() as stderr,
    ):
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr) as proc:
            i, buf = 0, proc.stdout.read(size)
            while len(buf) == size:
                if i in cfg.verify_frames:
                    frames[i] = buf
                bar.update(1)
                i, buf = i + 1, proc.stdout.read(size)
        show(stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"exit code {proc.returncode} of the sequential decode")
    return frames


def episode_command(cfg: Access, video: Path) -> list[str]:
    """(2) of specs/perception-data.md: every decoded frame of `video` once, raw yuv420p,
    to stdout."""
    return (
        [cfg.ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(video)]
        + ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p"]
        + ["pipe:1"]
    )


def decode_episode(
    cfg: Access, video: Path, n: int, keep: Container[int], timeout: float
) -> dict[int, bytes] | None:
    """(2) of specs/perception-data.md: one whole decode, the frames i in `keep`; None
    unless it exits with code 0 and exactly n frames; TimeoutError when its timer killed
    ffmpeg after `timeout` s."""
    size, frames, i, fired = frame_bytes(cfg), {}, 0, threading.Event()
    with tempfile.TemporaryFile() as stderr:
        cmd = episode_command(cfg, video)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr) as proc:

            def kill() -> None:
                fired.set()
                proc.kill()

            timer = threading.Timer(timeout, kill)
            timer.start()
            try:
                while len(buf := proc.stdout.read(size)) == size:
                    if i in keep:
                        frames[i] = buf
                    i += 1
                proc.wait()
            finally:
                timer.cancel()
        show(stderr)
    if fired.is_set() and proc.returncode == -signal.SIGKILL:
        raise TimeoutError(f"decode of {video} not ended within {timeout} s")
    return frames if proc.returncode == 0 and i == n and not buf else None
