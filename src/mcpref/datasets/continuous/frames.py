"""Bounded video decoding and byte-exact FFV1 shard payloads."""

import hashlib
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from mcpref.datasets.clips.episode import classify_frames, start_modes
from mcpref.datasets.vpt.video.ffmpeg import (
    _read_frame,
    episode_command,
    frame_bytes,
    show,
)
from mcpref.gui.actions import needed_frames
from mcpref.infrastructure.runs.files import write_json


class InvalidVideo(ValueError):
    """A fully attempted decode disagrees with the source row count."""


def decoded(cfg, path: Path, expected: int):
    """Yield original decoder bytes; reject the entire piece on a count/code mismatch."""
    with tempfile.TemporaryFile() as stderr:
        command = episode_command(cfg, path)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr) as proc:
            fired = threading.Event()

            def kill():
                fired.set()
                proc.kill()

            timer = threading.Timer(cfg.ffmpeg_timeout, kill)
            timer.start()
            count, size, buf = 0, 0, bytearray(frame_bytes(cfg))
            try:
                while (size := _read_frame(proc.stdout, buf)) == len(buf):
                    yield count, bytes(buf)
                    count += 1
                proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                timer.cancel()
        show(stderr)
    if fired.is_set() and proc.returncode == -signal.SIGKILL:
        raise TimeoutError(f"Video decode timed out: {path}")
    if proc.returncode or size or count != expected:
        raise InvalidVideo(
            f"Invalid decoded video: {count} frames, expected {expected}"
        )


def verify(pipe, target: Path, rows, fingerprint: dict) -> dict:
    path = target / "verified.json"
    if path.exists():
        import json

        old = json.loads(path.read_text())
        if old["fingerprint"] == fingerprint:
            return old
    needed = set(needed_frames(rows))
    needed.update(i for i in range(min(2, len(rows))) if rows[i]["isGuiOpen"])
    classes, batch, frame_zero = {}, {}, None
    try:
        for i, frame in decoded(pipe.cfg, target / "source.mp4", len(rows) + 1):
            if i == 0:
                frame_zero = frame
            if i in needed:
                batch[i] = frame
            if len(batch) >= pipe.cfg.classifier_batch:
                classify_frames(pipe, batch, list(batch), classes)
                batch.clear()
        classify_frames(pipe, batch, list(batch), classes)
    except InvalidVideo as error:
        result = {"reason": "video", "detail": str(error)}
    else:
        result = {
            "reason": None,
            "classes": classes,
            "initial_modes": start_modes(pipe, rows[0]["starts_session"], frame_zero),
        }
    result["fingerprint"] = fingerprint
    write_json(path, result)
    return result


def encode(cfg, settings, raw: Path, output: Path, count: int) -> dict:
    command = [
        cfg.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s:v",
        f"{cfg.width}x{cfg.height}",
        "-r",
        "1",
        "-i",
        str(raw),
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-threads",
        str(settings.codec_threads),
        str(output),
    ]
    subprocess.run(command, check=True, timeout=cfg.ffmpeg_timeout)
    source_hash, decoded_hash = hashlib.sha256(), hashlib.sha256()
    with raw.open("rb") as source:
        for _, frame in decoded(cfg, output, count):
            original = source.read(len(frame))
            if original != frame:
                raise ValueError("FFV1 round trip changed source pixels")
            source_hash.update(original)
            decoded_hash.update(frame)
        if source.read(1):
            raise ValueError("FFV1 round trip lost source frames")
    return {
        "codec": "ffv1",
        "frames": count,
        "width": cfg.width,
        "height": cfg.height,
        "decoded_sha256": decoded_hash.hexdigest(),
        "raw_sha256": source_hash.hexdigest(),
    }
