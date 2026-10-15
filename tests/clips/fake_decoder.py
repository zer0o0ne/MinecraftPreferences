"""Fake ffmpeg for the tests of specs/perception-data.md; argv: <ctl.json> <log> <program>
<ffmpeg arguments>. Run with `python -I -S`: nothing imported from the tests.

Decodes the fake video named by `-i` (a line `FAKEMP4 <stem>`, then raw yuv420p frames of
ctl["frame_size"] bytes) to stdout as ctl["behaviours"][stem][attempt] says (default "ok"):
"ok" every frame; "short" without the last frame; "long" with the last frame twice;
"partial" with the last half frame missing; "exit1" every frame and exit code 1; "slow"
every frame after ctl["sleep"] s. First writes ctl["stderr"][stem][attempt] (hex) to
stderr, if any. Logs the start (with argv and pid) and the end of each run."""

import json
import os
import sys
import time
from pathlib import Path


def log(path: str, record: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, (json.dumps(record) + "\n").encode())
    os.close(fd)


def attempt(path: str, key: str) -> int:
    """Earlier starts of `key` in the log."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return sum(r["event"] == "start" and r["key"] == key for r in records)


def decoded(frames: bytes, size: int, behaviour: str) -> bytes:
    return {
        "short": frames[:-size],
        "long": frames + frames[-size:],
        "partial": frames[: -size // 2],
    }.get(behaviour, frames)


def main(ctl_path: str, log_path: str, *argv: str) -> int:
    ctl, args = json.loads(Path(ctl_path).read_text()), list(argv[1:])
    head, frames = Path(args[args.index("-i") + 1]).read_bytes().split(b"\n", 1)
    key = head.decode().removeprefix("FAKEMP4 ")
    k = attempt(log_path, key)
    plan, noise = ctl["behaviours"].get(key, []), ctl["stderr"].get(key, [])
    behaviour = plan[k] if k < len(plan) else "ok"
    record = {"event": "start", "key": key, "argv": list(argv), "t": time.time()}
    log(log_path, record | {"pid": os.getpid()})
    if k < len(noise):
        sys.stderr.buffer.write(bytes.fromhex(noise[k]))
        sys.stderr.buffer.flush()
    if behaviour == "slow":
        time.sleep(ctl["sleep"])
    sys.stdout.buffer.write(decoded(frames, ctl["frame_size"], behaviour))
    sys.stdout.buffer.flush()
    code = int(behaviour == "exit1")
    log(log_path, {"event": "end", "key": key, "t": time.time(), "code": code})
    return code


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
