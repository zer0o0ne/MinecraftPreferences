"""Streaming H.264 encoding and atomic publication without overwriting files."""

import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mcpref.infrastructure.shards.store import make_dirs, sync
from mcpref.runtime.video.config import VideoConfig


def destination(source: Path, output: Path) -> Path:
    output = output.absolute()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 extension")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the sampling directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    return output


@contextmanager
def staged_output(output: Path) -> Iterator[Path]:
    make_dirs(output.parent)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".part", dir=output.parent, delete=False
    ) as file:
        staged = Path(file.name)
    published = False
    try:
        yield staged
        sync(staged)
        os.link(staged, output)
        published = True
        sync(output.parent)
    except BaseException:
        if published:
            output.unlink()
        raise
    finally:
        staged.unlink(missing_ok=True)


def command(cfg: VideoConfig, output: Path, shape: tuple[int, ...]) -> list[str]:
    height, width, _ = shape
    return [
        cfg.ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(cfg.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]


def _write(process: subprocess.Popen, frames: Iterable[np.ndarray], total: int) -> None:
    with tqdm(total=total, desc="gameplay video", unit="frame") as bar:
        for frame in frames:
            process.stdin.write(frame.tobytes(order="C"))
            bar.update(1)
    process.stdin.close()
    process.wait()


def encode(
    cfg: VideoConfig,
    output: Path,
    frames: Iterable[np.ndarray],
    shape: tuple[int, ...],
    total: int,
) -> None:
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command(cfg, output, shape),
            stdin=subprocess.PIPE,
            stderr=errors,
            stdout=subprocess.DEVNULL,
        )
        try:
            try:
                _write(process, frames, total)
            except BrokenPipeError:
                process.wait()
                errors.seek(0)
                detail = errors.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg closed its input early: {detail}") from None
            if process.returncode:
                errors.seek(0)
                detail = errors.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited {process.returncode}: {detail}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
