"""Real FFmpeg round trips; run with ffmpeg on PATH or MCPREF_TEST_FFMPEG."""

import dataclasses
import os
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from mcpref.runtime.video import encoder
from mcpref.runtime.video.export import export
from runtime.video.test_export import configured


@pytest.fixture
def ffmpeg():
    executable = os.environ.get("MCPREF_TEST_FFMPEG") or shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("real FFmpeg is unavailable")
    return executable


@pytest.mark.parametrize(
    "rate,duration,time_base", [(10, "00.50", "1/10"), (2.5, "02.00", "2/5")]
)
def test_real_video(source, tmp_path, cfg, ffmpeg, rate, duration, time_base):
    cfg = dataclasses.replace(cfg, ffmpeg=ffmpeg, fps=Fraction(str(rate)))
    output = tmp_path / "clip.mp4"
    export(configured(cfg, source, output, start=1, stop=5))
    decoded = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(output),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    frames = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, 24, 32, 3)
    assert len(frames) == 5
    np.testing.assert_allclose(
        frames.mean(axis=(1, 2, 3)), [20, 40, 60, 80, 100], atol=3
    )
    info = decoded.stderr.decode()
    assert "Video: h264" in info and f"Duration: 00:00:{duration}" in info
    assert "yuv420p" in info and "Audio:" not in info
    metadata = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "framemd5", "pipe:1"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"#tb 0: {time_base}" in metadata
    assert (
        len([line for line in metadata.splitlines() if not line.startswith("#")]) == 5
    )
    container = output.read_bytes()
    assert container.index(b"moov") < container.index(b"mdat")


def test_progress_counts_both_endpoints(source, tmp_path, cfg, monkeypatch):
    from runtime.video.test_export import Process

    process = Process()
    bars = []

    class Bar:
        def __init__(self, **kwargs):
            self.total, self.count = kwargs["total"], 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            self.count += count

    monkeypatch.setattr(encoder, "tqdm", Bar)
    monkeypatch.setattr(encoder.subprocess, "Popen", lambda *args, **kwargs: process)
    export(configured(cfg, source, tmp_path / "clip.mp4", start=2, stop=4))
    assert [(bar.total, bar.count) for bar in bars] == [(3, 3)]
