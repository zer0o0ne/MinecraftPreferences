"""Checks of specs/vpt-gui-frames.md: the fake data of `main`, the ffmpeg commands it must
run, the byte ranges of (f), a config of `plan`, what `plan` returns and a reader of plan
shards."""

import collections
import json
import os
from pathlib import Path

from vpt_world import FIELDS, Ep, joined, options

from mcpref.vpt.gui_frames import GuiFramesConfig

EPISODES = {  # for `main`: at most 2 frames per episode, 10 frames in all
    "6xx": [Ep("data/6.9/old6.mp4", reason="version"), Ep("data/6.12/a6.mp4", gui=".###.##")],
    "7xx": [Ep("data/7.1/b7.mp4", gui="..####"),
            Ep("data/7.2/m7.mp4", reason="missing", options=None)],
    "8xx": [Ep("data/8.0/n8.mp4", gui="#.#.#", reason="no_gui"),
            Ep("data/8.1/c8.mp4", gui="##.##")],
    "9xx": [Ep("data/9.0/o9.mp4", gui=".##", reason="options", options=options(guiScale=3)),
            Ep("data/9.1/d9.mp4", gui="i#..##")],
    "10xx": [Ep("data/10.0/e10.mp4", gui="..######"),
             Ep(FIELDS["verify_relpath"], gui="." * 24, reason="no_gui")],
}  # fmt: skip


def stem(row: dict) -> str:
    """Stem of a planned frame's relpath."""
    return row["relpath"].rsplit("/", 1)[1].removesuffix(".mp4")


def key(row: dict) -> str:
    """Fake ffmpeg key `<stem>_<i>` of a planned frame."""
    return f"{stem(row)}_{row['frame']}"


def seek_time(i: int) -> str:
    """(4), transcribed: exact decimal of i / 20."""
    return f"{i // 20}.{5 * (i % 20):02d}"


def part_ranges(size: int, q: int) -> list[str]:
    """(f), transcribed: the Range values of a file of size L > 0 on q threads, p = ceil(L/q),
    part j = bytes [j p, min((j + 1) p, L) - 1], j < ceil(L / p)."""
    p = -(-size // q)
    return [f"bytes={j * p}-{min((j + 1) * p, size) - 1}" for j in range(-(-size // p))]


def command_head(fields: dict, url: str, i: int) -> list[str]:
    """`ffmpeg_command` of Interface without its last item, the output path."""
    micro = str(int(fields["ffmpeg_rw_timeout"] * 10**6))
    header = f"x-ms-version: {fields['x_ms_version']}\r\n"
    return [fields["ffmpeg"], "-nostdin", "-loglevel", "error", "-y", "-headers", header,
            "-rw_timeout", micro, "-ss", seek_time(i), "-i", url, "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "yuv420p"]  # fmt: skip


def plan_config(world, **changes) -> GuiFramesConfig:
    """The config of `plan` on `world` (its index_urls; ffmpeg and out_dir unused)."""
    fields = FIELDS | {
        "index_urls": world.index_urls,
        "ffmpeg": "ffmpeg",
        "out_dir": "o",
    }
    return GuiFramesConfig(**(fields | changes))


def totals(shards: list[dict]) -> tuple[int, collections.Counter]:
    """What `plan` returns for `shards`: frames planned, episodes per reason."""
    reasons = collections.Counter(r["reason"] for r in joined(shards, "episodes"))
    return len(joined(shards, "frames")), reasons


def read_shards(directory: Path) -> list[dict]:
    """The complete plan shards of `directory` in order: entries named name(k) = %06d for
    k = 0, ..., n - 1 (staged `.part` entries ignored), each holding only `shard.json`."""
    names = [] if not directory.exists() else sorted(os.listdir(directory))
    names = [x for x in names if not x.endswith(".part")]
    assert names == [f"{k:06d}" for k in range(len(names))], names
    assert all(os.listdir(directory / x) == ["shard.json"] for x in names), names
    return [json.loads((directory / x / "shard.json").read_text()) for x in names]
