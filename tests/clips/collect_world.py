"""Fake VPT episodes for the tests of specs/perception-data.md: `.jsonl` lines from a
script of one character per line, raw yuv420p frames of known content and fake videos;
a tiny catalogue (specs/gui-screens.md forms) and a font with one glyph.

Script characters: `w` a world line holding W; `q` W and Q (a foreign line); `f` W and F3
(the overlay toggles one frame after the release); `v` W, the pitch 12 degrees higher from
this line on; `i` the player inventory; `p` the player inventory with a left press over
slot 1; `c` a container. Every line: yaw 0, pitch 0.01 t (plus the jumps of `v`), stats
empty only on line 0 of an episode that starts a session.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from gui.gui_action_lines import S1, box, episode, inv, world
from PIL import Image

W, H = 8, 12  # frame size; the overlay box of a one-glyph prefix lies in rows 1..9
FRAME = W * H * 3 // 2
ROWS = {"w": world(keys=["w"]), "q": world(keys=["w", "q"]),
        "f": world(keys=["w", "f3"]), "v": world(keys=["w"]), "i": inv(),
        "p": inv(press=[0], at=S1), "c": box()}  # fmt: skip
GUI = "ipc"
PLAY = "minecraft.custom:minecraft.play_time"
GOOD = "good"  # the well-formed file
REQUIRED = {"guiScale": 2, "autoJump": False, "key_key.forward": "key.keyboard.w"}
LUMA = {True: 200, False: 16}  # frame 0 above its last row: overlay shown or not
TEXTURES = {"url": "http://127.0.0.1:9/c.jar", "sha1": "0" * 40, "http_timeout": 1.0,
            "http_retries": 0, "retry_delay": 0.0}  # fmt: skip
CATALOGUE = {name: {"description": f"{name}.", "window": [90, 40, 200, 120],
                    "members": {name: {"blits": [], "dynamic": [], "labels": []}},
                    "elements": [{"index": i, "name": f"e{i}", "kind": kind, "hit": hit,
                                  "click": [hit[0] + hit[2] / 2, hit[1] + hit[3] / 2]}
                                 for i, kind, hit in elements]}
             for name, elements in {
                 "inv": [(1, "slot", [100, 100, 18, 18])],
                 "box": [(1, "slot", [100, 100, 18, 18]), (37, "slot", [100, 50, 18, 18])],
                 "pause": [(37, "button", [100, 100, 40, 20])]}.items()}  # fmt: skip


def lines(script: str, session: bool) -> list[dict]:
    out = episode(*(ROWS[c] for c in script))
    jumps = np.cumsum([c == "v" for c in script])
    for t, line in enumerate(out):
        line["pitch"] = 0.01 * t + 12.0 * int(jumps[t])
        line["stats"] = {} if session and t == 0 else {PLAY: 100 + t}
        line["keyboard"]["chars"] = "é"
    return out


def jsonl(script: str, session: bool) -> bytes:
    """Compact separators and raw UTF-8: re-serialising a line changes its bytes."""
    dumps = [json.dumps(x, separators=(",", ":"), ensure_ascii=False) for x in lines(script, session)]  # fmt: skip
    return "".join(d + "\n" for d in dumps).encode()


def options(**changes) -> bytes:
    """`lastServer` is not ASCII: a JSON file of the block escapes it."""
    other = {"fov": 0.5, "lastServer": "café"}
    return json.dumps(REQUIRED | other | changes, indent=1).encode()


def frame(stem: str, t: int, overlay: bool) -> bytes:
    """Random planes; frame 0 has a uniform luma above its last row (LUMA)."""
    digest = hashlib.sha256(f"{stem}/{t}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    y = rng.integers(16, 236, (H, W))
    if t == 0:
        y[: H - 1] = LUMA[overlay]
    u, v = rng.integers(16, 241, (2, H // 2, W // 2))
    return b"".join(p.astype(np.uint8).tobytes() for p in (y, u, v))


@dataclass(frozen=True)
class Ep:
    """An episode; `options`, `jsonl`, `video`: served bytes, None (404) or GOOD;
    `classes`: line -> class answered by the fake classifier instead of `inv` (`box` for
    `c` lines)."""

    relpath: str
    script: str = "w" * 12
    session: bool = False
    overlay: bool = False
    classes: dict = field(default_factory=dict)
    options: object = GOOD
    jsonl: object = GOOD
    video: object = GOOD

    @property
    def version(self) -> str:
        return self.relpath.split("/")[1]

    @property
    def stem(self) -> str:
        return self.relpath.split("/")[2].removesuffix(".mp4")

    def frames(self) -> list[bytes]:
        """Frames 0..T."""
        return [frame(self.stem, t, self.overlay) for t in range(len(self.script) + 1)]

    def files(self) -> dict:
        """Suffix next to the stem -> served bytes, or None for 404."""
        good = {
            "-options.json": options(),
            ".jsonl": jsonl(self.script, self.session),
            ".mp4": f"FAKEMP4 {self.stem}\n".encode() + b"".join(self.frames()),
        }
        given = {
            "-options.json": self.options,
            ".jsonl": self.jsonl,
            ".mp4": self.video,
        }
        return {k: good[k] if v is GOOD else v for k, v in given.items()}

    def cls(self, t: int) -> str:
        """The class the fake classifier gives frame t of a GUI line."""
        return self.classes.get(t, "box" if self.script[t] == "c" else "inv")


BAD_JSONL = jsonl("wwi", False) + b'{"isGuiOpen": tr\n'
EPISODES = {  # with delta 2, 3 states, 2 clips per episode
    "6xx": [Ep("data/6.9/old6.mp4"),
            Ep("data/6.12/a6.mp4", "wwwwiiwwwwpwwwwcccwwwwwiiiwwwwww", classes={24: "pause"})],
    "7xx": [Ep("data/7.1/b7.mp4", "iiwwwwwwvwwwwwwwwwiiwwwwww", session=True, overlay=True,
               classes={18: "other"}),
            Ep("data/7.2/m7.mp4", options=None),
            Ep("data/7.3/ov7.mp4", overlay=True),
            Ep("data/7.1/b7.mp4", "iiwwwwwwvwwwwwwwwwiiwwwwww", session=True, overlay=True,
               classes={18: "other"})],
    "8xx": [Ep("data/8.0/e8.mp4", jsonl=b""),
            Ep("data/8.1/c8.mp4", "wwwwfwwwwwwwwwwwwwwppwwwwww", overlay=True),
            Ep("data/8.2/x8.mp4", jsonl=BAD_JSONL)],
    "9xx": [Ep("data/9.0/o9.mp4", options=options(autoJump=True)),
            Ep("data/9.1/v9.mp4", video=None),
            Ep("data/9.2/n9.mp4", "wwww")],
    "10xx": [Ep("data/10.0/d10.mp4", "wwwwwqwwwwwwccwwwwwwwiwwwiipwwwwwww", classes={27: "other"}),
             Ep("data/10.0/r10.mp4", "wiiwiiwiiw", classes={1: "other", 4: "other", 7: "other"})],
}  # fmt: skip


def font(path: Path) -> None:
    """font/ascii.png with one texel, at (0, 0) of the cell of `A`: w = 1, adv = 2."""
    image = np.zeros((128, 128, 4), np.uint8)
    image[8 * (ord("A") // 16), 8 * (ord("A") % 16)] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
