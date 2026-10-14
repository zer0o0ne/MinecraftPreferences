"""Small catalogue in the YAML forms of specs/gui-screens.md (Interface); the approved one."""

import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml


def blit(texture: str, u: int, v: int, w: int, h: int, x: int, y: int) -> dict:
    return {"texture": texture, "u": u, "v": v, "w": w, "h": h, "x": x, "y": y}


def element(index: int, name: str, kind: str, hit: list, click: list) -> dict:
    return {"index": index, "name": name, "kind": kind, "hit": hit, "click": click}


@pytest.fixture
def catalogue() -> dict:
    """`alpha`: book panel, abutting slots 1-2, odd-sized button 37, labels of two colours;
    `beta`: no book panel."""
    first = {
        "blits": [
            blit("gui/container/inventory.png", 0, 0, 176, 166, 232, 97),
            blit("gui/widgets.png", 0, 66, 200, 20, 220, 200),
        ],
        "dynamic": [{"name": "arrow", "rect": [40, 2, 24, 17]}],
        "labels": [{"name": "Title", "rect": [8, 6, 31, 8], "color": 0x404040}],
    }
    second = {
        "blits": [],
        "dynamic": [],
        "labels": [{"name": "T", "rect": [8, 6, 5, 9], "color": 0xE0E0E0}],
    }
    alpha = {
        "members": {"first": first, "second": second},
        "description": "Alpha screen.",
        "window": [0, 0, 176, 166],
        "book_panel": [160, 97, 147, 166],
        "elements": [
            element(1, "a", "slot", [0, 0, 18, 18], [9.0, 9.0]),
            element(2, "b", "slot", [18, 0, 18, 18], [27.0, 9.0]),
            element(37, "c", "button", [5, 30, 21, 13], [15.5, 36.5]),
        ],
    }
    beta = {
        "members": {
            "only": {
                "blits": [blit("gui/book.png", 0, 0, 192, 192, 224, 2)],
                "dynamic": [],
                "labels": [],
            }
        },
        "description": "Beta screen.",
        "window": [224, 2, 192, 192],
        "elements": [
            element(37, "done", "button", [220, 200, 200, 20], [320.0, 210.0])
        ],
    }
    textures = {
        "url": "http://127.0.0.1:9/client.jar",
        "sha1": "0" * 40,
        "out_dir": "data/textures",
        "http_timeout": 2.0,
        "http_retries": 2,
        "retry_delay": 0.05,
    }
    return {"classes": {"alpha": alpha, "beta": beta}, "textures": textures}


@pytest.fixture(scope="session")
def classes(repo) -> dict:
    """`classes` of the approved configs/gui-screens.yaml as plain YAML; read-only."""
    text = (repo / "configs/gui-screens.yaml").read_text()
    return yaml.load(text, Loader=yaml.CSafeLoader)["classes"]


@pytest.fixture
def write_config(tmp_path):
    """Writes a catalogue dict as YAML; returns its path."""

    def write(data: dict) -> Path:
        path = tmp_path / "gui-screens.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return write


# specs/gui-labels.md: fake frames (specs/vpt-gui-frames.md), fake ask_many (specs/vlm-client.md).
FRAME_W, FRAME_H = 4, 2  # width, height in the fake frames' run.json
OUT_FILES = ("labels.jsonl", "summary.json", "run.json")


def lit(p: int) -> np.ndarray:
    """Fake frame p: pixel (r, c) is lit iff bit r * width + c of (37 p + 11) mod 256 is set."""
    bits = (37 * p + 11) % 256
    rows = [
        [bits >> (r * FRAME_W + c) & 1 for c in range(FRAME_W)] for r in range(FRAME_H)
    ]
    return np.array(rows, dtype=np.uint8)


def frame_yuv(p: int) -> bytes:
    """Raw yuv420p: Y plane (height x width) 235 where lit, else 16; U, V (height/2 x width/2) 128."""
    chroma = bytes([128]) * (lit(p).size // 2)
    return (16 + 219 * lit(p)).astype(np.uint8).tobytes() + chroma


def frame_rgb(p: int) -> np.ndarray:
    """vpt-gui-frames (5) by hand: U = V = 128 gives R = G = B = 255/219 (Y - 16): 16 -> 0, 235 -> 255."""
    return np.repeat(lit(p)[..., None] * np.uint8(255), 3, axis=2)


@pytest.fixture
def write_frames():
    """Writes a {frames_dir}: frames (status, line.isGuiInventory) by position, prev_line with the
    opposite flag, a .yuv for `ok` frames only, run.json with the frame size; returns the ids."""

    def write(root: Path, frames: list) -> list:
        shutil.rmtree(root, ignore_errors=True)
        (root / "frames").mkdir(parents=True)
        ids = [f"7.0_ep{p}_{p + 3}" for p in range(len(frames))]
        recs = []
        for p, (status, flag) in enumerate(frames):
            line = {"isGuiOpen": True, "isGuiInventory": flag}
            recs.append(
                {"id": ids[p], "subset": "7xx", "relpath": f"data/7.0/ep{p}.mp4"}
            )
            recs[-1].update(version="7.0", frame=p + 3, status=status, line=line)
            recs[-1]["prev_line"] = {**line, "isGuiInventory": not flag}
            if status == "ok":
                (root / "frames" / f"{ids[p]}.yuv").write_bytes(frame_yuv(p))
        (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        counts = dict(Counter(s for s, _ in frames))
        config = {"width": FRAME_W, "height": FRAME_H, "fps": 20}
        run = {"config": config, "index_etags": {}, "episodes": {}, "frames": counts}
        (root / "run.json").write_text(json.dumps(run))
        return ids

    return write


@dataclass
class Call:
    prompts: list
    seeds: list
    cfg: object
    files: dict  # output file name -> (text, handle opened at call time)


class FakeAskMany:
    """ask_many: the answer to a prompt is the next of answers[p], p the frame whose RGB view is
    its image (None if none is); raises on call `fail_on`; records calls and the output files."""

    def __init__(self, answers: dict, out: Path):
        self.answers, self.out, self.fail_on, self.calls = answers, out, None, []
        self.asked = Counter()
        self.images = {self.key(frame_rgb(p)): p for p in answers}

    @staticmethod
    def key(image) -> tuple | None:
        if not isinstance(image, np.ndarray):
            return None
        return image.shape, image.dtype.str, image.tobytes()

    def position(self, prompt) -> int | None:
        return self.images.get(self.key(list(prompt)[1])) if len(prompt) > 1 else None

    def __call__(self, prompts, seeds, cfg) -> list:
        paths = [self.out / n for n in OUT_FILES if (self.out / n).exists()]
        files = {p.name: (p.read_text(), p.open()) for p in paths}
        self.calls.append(Call([list(q) for q in prompts], list(seeds), cfg, files))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("HTTP 400: rejected")
        return [self.answer(self.position(q)) for q in prompts]

    def answer(self, p: int | None) -> str | None:
        if p is None:
            return None
        self.asked[p] += 1
        return self.answers[p][min(self.asked[p], len(self.answers[p])) - 1]


@pytest.fixture
def fake_ask_many(monkeypatch):
    """Factory: a FakeAskMany put in place of ask_many in every mcpref module holding it
    (mcpref.gui.labels is imported first, so the modules it loads are among them)."""
    import mcpref.gui.labels
    import mcpref.vlm.client

    real, fakes = mcpref.vlm.client.ask_many, []

    def make(answers: dict, out: Path) -> FakeAskMany:
        fakes.append(FakeAskMany(answers, out))
        for name, module in list(sys.modules.items()):
            if name.startswith("mcpref") and getattr(module, "ask_many", None) is real:
                monkeypatch.setattr(module, "ask_many", fakes[-1])
        return fakes[-1]

    yield make
    for call in (c for fake in fakes for c in fake.calls):
        for _, handle in call.files.values():
            handle.close()
