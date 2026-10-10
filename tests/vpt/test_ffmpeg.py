"""(4) and `ffmpeg_command` of specs/vpt-gui-frames.md (Interface)."""

import re
from decimal import Decimal
from pathlib import Path

import pytest
from vpt_world import FIELDS

from mcpref.vpt.gui_frames import GuiFramesConfig, ffmpeg_command, frame_time


@pytest.mark.parametrize(
    "i, t",
    [(0, "0.00"), (1, "0.05"), (2, "0.10"), (19, "0.95"), (20, "1.00"), (21, "1.05"),
     (1271, "63.55"), (5999, "299.95"), (6000, "300.00")],
)  # fmt: skip
def test_frame_time_hand_cases_eq4(i, t):
    assert frame_time(i, 20) == t


def test_frame_time_is_exact_decimal_of_i_over_20():
    """[R]: t is the exact decimal of i / 20, two decimals, for every i of a 2-hour video."""
    for i in range(20 * 7200):
        t = frame_time(i, 20)
        assert re.fullmatch(r"\d+\.\d\d", t) and Decimal(t) == Decimal(i) / 20, (i, t)


def config(**changes) -> GuiFramesConfig:
    fields = FIELDS | {"index_urls": {}, "ffmpeg": "/opt/ff/bin/ffmpeg", "out_dir": "o"}
    return GuiFramesConfig(**(fields | changes))


@pytest.mark.parametrize(
    "rw_timeout, micro", [(7, "7000000"), (60, "60000000"), (2.5, "2500000")]
)
def test_ffmpeg_command_is_the_interface_list(rw_timeout, micro):
    cfg = config(x_ms_version="2019-12-12", ffmpeg_rw_timeout=rw_timeout)
    url = "https://blob.test/minecraft-rl/data/6.13/Player1-f153-20211211-220855.mp4"
    out = Path("/data/out/frames/6.13_Player1-f153-20211211-220855_1271.yuv.part")
    assert ffmpeg_command(cfg, url, 1271, out) == [
        "/opt/ff/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-headers", "x-ms-version: 2019-12-12\r\n",
        "-rw_timeout", micro, "-ss", "63.55", "-i", url,
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", str(out),
    ]  # fmt: skip


def test_ffmpeg_command_takes_header_version_from_config():
    cmd = ffmpeg_command(
        config(x_ms_version="2020-04-08"), "http://h/v.mp4", 0, Path("o")
    )
    assert cmd[cmd.index("-headers") + 1] == "x-ms-version: 2020-04-08\r\n"
    assert cmd[cmd.index("-ss") + 1] == "0.00"
