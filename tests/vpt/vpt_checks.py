"""Checks of `main` in specs/vpt-gui-frames.md: its fake data and the ffmpeg commands it
must run."""

from vpt_world import FIELDS, Ep, options

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
    """Stem of a frames.jsonl row's relpath."""
    return row["relpath"].rsplit("/", 1)[1].removesuffix(".mp4")


def key(row: dict) -> str:
    """Fake ffmpeg key `<stem>_<i>` of a frames.jsonl row."""
    return f"{stem(row)}_{row['frame']}"


def seek_time(i: int) -> str:
    """(4), transcribed: exact decimal of i / 20."""
    return f"{i // 20}.{5 * (i % 20):02d}"


def command_head(fields: dict, url: str, i: int) -> list[str]:
    """`ffmpeg_command` of Interface without its last item, the output path."""
    micro = str(int(fields["ffmpeg_rw_timeout"] * 10**6))
    header = f"x-ms-version: {fields['x_ms_version']}\r\n"
    return [fields["ffmpeg"], "-nostdin", "-loglevel", "error", "-y", "-headers", header,
            "-rw_timeout", micro, "-ss", seek_time(i), "-i", url, "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "yuv420p"]  # fmt: skip
