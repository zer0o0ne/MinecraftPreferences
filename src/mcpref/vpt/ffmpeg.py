"""(4), (c), (e) of specs/vpt-gui-frames.md: frames extracted and decoded by ffmpeg."""

import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import BinaryIO

from tqdm import tqdm

from mcpref.http.retry import retried
from mcpref.vpt.config import GuiFramesConfig


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


def frame_bytes(cfg: GuiFramesConfig) -> int:
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
    """(c): ffmpeg into <path>.part, retried; renamed to `path` on success."""
    part = path.with_name(path.name + ".part")
    run = partial(ffmpeg_ok, cfg, url, i, part)
    ok = retried(run, lambda ok: not ok, cfg.http_retries, cfg.retry_delay)
    if ok:
        part.rename(path)
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
