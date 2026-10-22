"""Small catalogue in the YAML forms of specs/gui-screens.md (Interface); the approved one;
fake ask_many and fake run directory of specs/gui-labels.md."""

import copy
import os
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


# specs/gui-labels.md: fake ask_many (specs/vlm-client.md), the fake world of labels_world.py.
@dataclass
class Call:
    prompts: list
    seeds: list
    cfg: object
    run_json: str | None  # text of {out_dir}/run.json when called
    labels: list  # entries of {out_dir}/labels when called


class FakeAskMany:
    """ask_many: the answer to a prompt is the next of answers[p], p the frame whose image is the
    prompt's second part (None if none is); raises on call `fail_on` (counted over runs);
    records the calls and the outputs present at each."""

    def __init__(self, answers: dict, images: dict, out: Path):
        self.answers, self.out, self.fail_on, self.calls = answers, out, None, []
        self.asked = Counter()
        self.images = {self.key(image): p for p, image in images.items()}

    @staticmethod
    def key(image) -> tuple | None:
        if not isinstance(image, np.ndarray):
            return None
        return image.shape, image.dtype.str, image.tobytes()

    def position(self, prompt) -> int | None:
        return self.images.get(self.key(list(prompt)[1])) if len(prompt) > 1 else None

    def __call__(self, prompts, seeds, cfg) -> list:
        run, labels = self.out / "run.json", self.out / "labels"
        text = run.read_text() if run.exists() else None
        names = sorted(p.name for p in labels.iterdir()) if labels.exists() else []
        self.calls.append(
            Call([list(q) for q in prompts], list(seeds), cfg, text, names)
        )
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

    real = mcpref.vlm.client.ask_many

    def make(answers: dict, images: dict, out: Path) -> FakeAskMany:
        fake = FakeAskMany(answers, images, out)
        for name, module in list(sys.modules.items()):
            if name.startswith("mcpref") and getattr(module, "ask_many", None) is real:
                monkeypatch.setattr(module, "ask_many", fake)
        return fake

    return make


@pytest.fixture
def lab(tmp_path, repo, catalogue, write_config, fake_ask_many, monkeypatch, capsys):
    """Catalogue inventory, inventory_book, chest3; FRAMES in SHARDS, K = 2; a vlm.yaml copy whose
    server refuses (an unfaked ask_many fails); config paths relative to the repository root;
    cwd elsewhere; tqdm's defaults (no TQDM_* variable)."""
    from labels_world import (
        FRAMES,
        PATHS,
        SEED,
        SHARDS,
        TEXTS,
        Lab,
        frame_rgb,
        write_frames,
    )
    from progress_bars import tqdm_defaults

    alpha, beta = catalogue["classes"]["alpha"], catalogue["classes"]["beta"]
    book = {**copy.deepcopy(alpha), "description": "Alpha with the book."}
    catalogue["classes"] = {"inventory": alpha, "inventory_book": book, "chest3": beta}
    write_config(catalogue)
    vlm = yaml.safe_load((repo / "configs/vlm.yaml").read_text())
    vlm["client"].update(base_url="http://127.0.0.1:9", max_resends=0)
    (tmp_path / "vlm.yaml").write_text(yaml.safe_dump(vlm))
    statuses = [f[:2] for f in FRAMES]
    ids = write_frames(tmp_path / "frames", statuses, SHARDS, 2, True)
    paths = {k: os.path.relpath(tmp_path / v, repo) for k, v in PATHS.items()}
    cfg = {**paths, **TEXTS, "seed": SEED}
    (tmp_path / "gui-labels.yaml").write_text(yaml.safe_dump(cfg))
    answers = {p: [a for a, _ in f[2]] for p, f in enumerate(FRAMES) if f[0] == "ok"}
    fake = fake_ask_many(answers, {p: frame_rgb(p) for p in answers}, tmp_path / "out")
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    tqdm_defaults(monkeypatch)
    return Lab(tmp_path, cfg, ids, fake, capsys)


# specs/action-mask.md: the approved catalogue, read-only (the masks: tests/conftest.py).
@pytest.fixture(scope="session")
def approved_screens(repo) -> dict:
    """`load_screens` of the approved configs/gui-screens.yaml."""
    from mcpref.gui.screens import load_screens

    return load_screens(repo / "configs/gui-screens.yaml")
