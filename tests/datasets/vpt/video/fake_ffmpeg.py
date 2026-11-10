"""Fake ffmpeg for tests/vpt; argv: <ctl.json> <log> <program> <ffmpeg arguments>.

A video is raw frames of ctl["frame_size"] bytes. With `-ss t`: GETs the `-i` URL with the
`-headers` lines and writes frame round(t * fps) to the last argument, as the behaviour
ctl["behaviours"][key][attempt] says (key `<stem>_<i>`; default "ok"). Without `-ss`: copies
the `-i` file to the last argument (`-`, `pipe:`, `pipe:1`: stdout). First writes the bytes
ctl["stderr"][key][attempt] (hex) to stderr, if any. Logs start and end."""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def log(path: str, record: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, (json.dumps(record) + "\n").encode())
    os.close(fd)


def attempt(path: str, key: str) -> int:
    """Earlier starts of `key` in the log."""
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return sum(r["event"] == "start" and r["key"] == key for r in records)


def arg(args: list, flag: str) -> str:
    return args[args.index(flag) + 1]


def extract(ctl: dict, args: list, behaviour: str, i: int) -> int:
    lines = arg(args, "-headers").split("\r\n")
    headers = dict(h.split(": ", 1) for h in lines if h)
    request = urllib.request.Request(arg(args, "-i"), headers=headers)
    try:
        data = urllib.request.urlopen(request, timeout=10).read()
    except OSError:
        return 1
    size = ctl["frame_size"]
    frame = data[i * size : (i + 1) * size]
    if behaviour in ("slow", "blocked"):
        time.sleep(ctl["sleep"])
    if behaviour == "blocked":  # a non-empty directory where the frame is renamed to
        Path(args[-1].removesuffix(".part"), "x").mkdir(parents=True)
    frame = {
        "wrong": bytes(255 - b for b in frame),
        "short": frame[:-1],
        "long": frame + b"\0",
    }.get(behaviour, frame)
    Path(args[-1]).write_bytes(frame)
    return int(behaviour == "exit1")


def decode(args: list) -> int:
    data = Path(arg(args, "-i").removeprefix("file:")).read_bytes()
    if args[-1] in ("-", "pipe:", "pipe:1"):
        sys.stdout.buffer.write(data)
    else:
        Path(args[-1]).write_bytes(data)
    return 0


def main(ctl_path: str, log_path: str, *argv: str) -> int:
    ctl, args = json.loads(Path(ctl_path).read_text()), list(argv[1:])
    stem = arg(args, "-i").rsplit("/", 1)[-1].removesuffix(".mp4")
    i = round(float(arg(args, "-ss")) * ctl["fps"]) if "-ss" in args else None
    key = "decode" if i is None else f"{stem}_{i}"
    plan = ctl["behaviours"].get(key, [])
    k = attempt(log_path, key) if os.path.exists(log_path) else 0
    behaviour = plan[k] if k < len(plan) else "ok"
    log(log_path, {"event": "start", "key": key, "argv": list(argv), "t": time.time()})
    noise = ctl["stderr"].get(key, [])
    if k < len(noise):
        sys.stderr.buffer.write(bytes.fromhex(noise[k]))
        sys.stderr.buffer.flush()
    code = decode(args) if i is None else extract(ctl, args, behaviour, i)
    log(log_path, {"event": "end", "key": key, "t": time.time(), "code": code})
    return code


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
