"""Fake world of `main` of specs/gui-labels.md: a {frames_dir} written as specs/vpt-gui-frames.md
(Produces) says, frames with their answers, expected label shards, the states of the bar `label`."""

import itertools
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from progress_bars import Bar, parse, screen, states

from mcpref.gui.labels import main

FRAME_W, FRAME_H = 4, 2  # width, height in the fake frames' run.json
PATHS = {"frames_dir": "frames", "screens": "gui-screens.yaml", "vlm": "vlm.yaml"}
PATHS["out_dir"] = "out"
TEXTS = {"instruction": "Screen?", "question": "Which?\n{classes}"}
TEXTS.update(class_line="{name}: {description}", other_description="something else")
QUESTION = "Which?\ninventory: Alpha screen.\ninventory_book: Alpha with the book."
QUESTION += "\nchest3: Beta screen.\nother: something else"
FRAMES = [  # by position: status, line.isGuiInventory, (answer, its label by (3)) per ask
    ("failed", True, []),
    ("ok", True, [("inventory", "inventory")]),
    ("ok", False, [(" `inventory_book` \n", "inventory_book")]),
    ("failed", False, []),
    ("ok", True, [("chest3", "chest3")]),
    ("ok", False, [("other", "other")]),
    ("ok", True, [("Chest3", "invalid")]),
    ("ok", False, [(None, "no_answer"), ("'chest3'", "chest3")]),
    ("ok", True, [('"other"', "other")]),
    ("failed", True, []),
]
# plan shard k: positions of its frames; label shard k: `end` after them
SHARDS, ENDS = [[0, 1, 2], [], [3], [4, 5, 6, 7], [8, 9]], [3, 3, 4, 8, 10]
NAMES = [f"{k:06d}" for k in range(len(SHARDS))]  # name(k) of specs/shards.md
FINGERPRINT = {"shards": 5, "ok": 7, "failed": 3}
OKS = FINGERPRINT["ok"]  # ok frames: the bar's total
SEED = 5
HASHED = {"config": "gui-labels.yaml", "vlm": "vlm.yaml", "screens": "gui-screens.yaml"}
VPT = [f"{y}_flag_{g}" for y in ("inventory", "other") for g in ("true", "false")]
# labels of the ok frames, each asked once
LABELS = ["inventory", "inventory_book", "chest3", "other", "invalid"]
LABELS += ["no_answer", "other"]
OPTIONS = {"guiScale": 2, "lang": "en_us"}  # a parsed options file


def lit(p: int) -> np.ndarray:
    """Frame p: pixel (r, c) is lit iff bit r * width + c of (37 p + 11) mod 256 is set."""
    bits = (37 * p + 11) % 256
    rows = [
        [bits >> (r * FRAME_W + c) & 1 for c in range(FRAME_W)] for r in range(FRAME_H)
    ]
    return np.array(rows, dtype=np.uint8)


def frame_yuv(p: int) -> bytes:
    """Raw yuv420p: Y (height x width) 235 where lit, else 16; U, V (height/2 x width/2) 128."""
    chroma = bytes([128]) * (lit(p).size // 2)
    return (16 + 219 * lit(p)).astype(np.uint8).tobytes() + chroma


def frame_rgb(p: int) -> np.ndarray:
    """vpt-gui-frames (5) by hand: U = V = 128, so R = G = B = 255/219 (Y - 16):
    16 -> 0, 235 -> 255."""
    return np.repeat(lit(p)[..., None] * np.uint8(255), 3, axis=2)


def episode(subset: str, relpath: str, reason: str | None) -> dict:
    """An episode record of a plan shard; options parsed for valid and `no_gui` episodes."""
    return {"subset": subset, "relpath": relpath, "reason": reason, "options": OPTIONS}


def plan_shard(frames: list, ids: list, ps: list, k: int) -> dict:
    """shard.json: one valid episode per frame, an invalid draw for a shard without frames;
    line.isGuiInventory = the frame's flag, prev_line the opposite; a stand-in rng state."""
    episodes, recs = [], []
    for p in ps:
        relpath, flag = f"data/7.0/ep{p}.mp4", frames[p][1]
        episodes.append(episode("7xx", relpath, None))
        line = {"isGuiOpen": True, "isGuiInventory": flag}
        recs.append({"id": ids[p], "subset": "7xx", "relpath": relpath})
        recs[-1].update(version="7.0", frame=p + 3)
        recs[-1].update(prev_line={**line, "isGuiInventory": not flag}, line=line)
    if not ps:
        episodes.append(episode("6xx", f"data/6.13/bare{k}.mp4", "no_gui"))
    rng = {"bit_generator": "PCG64", "state": {"state": 7 + k, "inc": 1}}
    rng.update(has_uint32=0, uinteger=0)
    return {"episodes": episodes, "frames": recs, "rng": rng}


def write_frames(
    root: Path, frames: list, shards: list, draws: int, counts: bool
) -> list:
    """{frames_dir} afresh: plan shards of the positions in `shards`, a .yuv per `ok` frame,
    run.json with K = `draws`, the frame size and, if `counts`, the counts; returns the ids."""
    shutil.rmtree(root, ignore_errors=True)
    (root / "frames").mkdir(parents=True)
    ids = [f"7.0_ep{p}_{p + 3}" for p in range(len(frames))]
    for k, ps in enumerate(shards):
        (root / "plan" / f"{k:06d}").mkdir(parents=True)
        text = json.dumps(plan_shard(frames, ids, ps, k))
        (root / "plan" / f"{k:06d}" / "shard.json").write_text(text)
    for p in (p for p, f in enumerate(frames) if f[0] == "ok"):
        (root / "frames" / f"{ids[p]}.yuv").write_bytes(frame_yuv(p))
    config = {"width": FRAME_W, "height": FRAME_H, "fps": 20, "plan_shard_draws": draws}
    run = {"config": config}
    if counts:
        episodes = Counter("null" for ps in shards for _ in ps)
        run["episodes"] = dict(episodes + Counter("no_gui" for ps in shards if not ps))
        run["frames"] = dict(Counter(f[0] for f in frames))
    (root / "run.json").write_text(json.dumps(run))
    return ids


def ok_of(k: int) -> list:
    """Positions of the ok frames of plan shard k."""
    return [p for p in SHARDS[k] if FRAMES[p][0] == "ok"]


def calls(first: int = 0) -> list:
    """ask_many calls of plan shards first, ...: (frame positions, seeds by (2))."""
    oks = [ok_of(k) for k in range(first, len(SHARDS))]
    return [(ps, [SEED + p for p in ps]) for ps in oks if ps]


def summary(labels: list, *flags: int) -> dict:
    """summary.json: the count of each label listed; the four (5) counts in the order of VPT."""
    return {"labels": dict(Counter(labels)), "vpt_check": dict(zip(VPT, flags))}


def label_bar(n: int, total: int = OKS) -> Bar:
    """A state of the bar `label`: unit frame, no postfix."""
    return Bar("label", n, total, "frame")


def check_bar(err: str, initial: int) -> None:
    """stderr holds only the bar `label`, drawn once: from `initial`, never back, its total
    and unit kept, closed at the ok count on a line of its own."""
    assert all(parse(s) for s in re.split(r"[\r\n]", err) if s.strip(" ")), repr(err)
    drawn = states(err, "label")
    assert drawn[0] == label_bar(initial), drawn
    kept = {(b.total, b.unit, b.postfix) for b in drawn}
    assert kept == {(OKS, "frame", None)}, drawn
    assert all(a.n <= b.n for a, b in itertools.pairwise(drawn)), drawn
    assert screen(err) == [label_bar(OKS)], repr(err)


@dataclass
class Lab:
    """One fake run directory; `run` calls main, keeps its stdout, returns (exit code, stderr)."""

    root: Path
    cfg: dict  # the config as written
    ids: list
    fake: object
    capsys: object
    out: str = ""

    def run(self) -> tuple[int, str]:
        code = main([str(self.root / "gui-labels.yaml")])
        self.out, err = self.capsys.readouterr()
        return code, err

    def frames(self, frames=FRAMES, shards=SHARDS, draws=2, counts=True) -> None:
        """Rewrites {frames_dir}."""
        statuses = [f[:2] for f in frames]
        write_frames(self.root / "frames", statuses, shards, draws, counts)

    def read(self, name: str):
        return json.loads((self.root / "out" / name).read_text())

    def shard(self, k: int):
        return self.read(f"labels/{k:06d}/labels.json")

    def names(self) -> list:
        """Entries of {out_dir}/labels."""
        labels = self.root / "out" / "labels"
        return sorted(p.name for p in labels.iterdir()) if labels.exists() else []

    def outputs(self) -> dict:
        """Every file under {out_dir}: relative path -> bytes."""
        out = self.root / "out"
        files = (p for p in out.rglob("*") if p.is_file())
        return {str(p.relative_to(out)): p.read_bytes() for p in files}

    def label_shard(self, k: int) -> dict:
        """labels.json of label shard k, its frames asked once; seeds by (2)."""
        recs = [{"id": self.ids[p], "seed": SEED + p} for p in ok_of(k)]
        for rec, p in zip(recs, ok_of(k)):
            rec.update(zip(("answer", "label"), FRAMES[p][2][0]))
            rec["gui_inventory"] = FRAMES[p][1]
        return {"labels": recs, "end": ENDS[k]}

    def asked(self, first: int = 0) -> list:
        """Per ask_many call from `first` on: (frame positions of its prompts, its seeds)."""
        made = self.fake.calls[first:]
        return [([self.fake.position(q) for q in c.prompts], c.seeds) for c in made]
