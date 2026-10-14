"""Fake run of `main` of specs/gui-labels.md: config texts, frames with their answers, expected
records, the states of the bar `label`."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from progress_bars import Bar

from mcpref.gui.labels import main

PATHS = {"frames_dir": "frames", "screens": "gui-screens.yaml", "vlm": "vlm.yaml"}
PATHS["out_dir"] = "out"
TEXTS = {"instruction": "Screen?", "question": "Which?\n{classes}"}
TEXTS.update(class_line="{name}: {description}", other_description="something else")
QUESTION = "Which?\ninventory: Alpha screen.\ninventory_book: Alpha with the book."
QUESTION += "\nchest3: Beta screen.\nother: something else"
FRAMES = [  # status, line.isGuiInventory, (answer, its label by (3)) of successive asks
    ("failed", True, []),
    ("ok", True, [("inventory", "inventory")]),
    ("ok", False, [(" `inventory_book` \n", "inventory_book")]),
    ("failed", False, []),
    ("ok", True, [("chest3", "chest3")]),
    ("ok", False, [("other", "other")]),
    ("ok", True, [("Chest3", "invalid")]),
    ("ok", False, [(None, "no_answer"), ("'chest3'", "chest3")]),
    ("ok", True, [('"other"', "other")]),
]
OK = [1, 2, 4, 5, 6, 7, 8]
SEED = 5
BATCHES = [[1, 2, 4], [5, 6, 7], [8]]  # batch = 3
VPT = [f"{y}_flag_{g}" for y in ("inventory", "other") for g in ("true", "false")]
HASHED = {"config": "gui-labels.yaml", "vlm": "vlm.yaml", "screens": "gui-screens.yaml"}


def label_bar(n: int, total: int) -> Bar:
    """A state of the bar `label`: unit frame, no postfix."""
    return Bar("label", n, total, "frame")


def summary(labels: list, *flags: int) -> dict:
    """summary.json: the count of each label listed; the four (5) counts in the order of VPT."""
    return {"labels": dict(Counter(labels)), "vpt_check": dict(zip(VPT, flags))}


def jsonl(text: str) -> list:
    return [json.loads(line) for line in text.splitlines()]


@dataclass
class Lab:
    """One fake run directory; `run` calls main, keeps its stdout, returns (exit code, stderr)."""

    root: Path
    ids: list
    fake: object
    capsys: object
    write_frames: object
    out: str = ""

    def run(self) -> tuple[int, str]:
        code = main([str(self.root / "gui-labels.yaml")])
        self.out, err = self.capsys.readouterr()
        return code, err

    def read(self, name: str):
        text = (self.root / "out" / name).read_text()
        return jsonl(text) if name.endswith(".jsonl") else json.loads(text)

    def outputs(self) -> dict:
        out = self.root / "out"
        return {p.name: p.read_bytes() for p in out.iterdir()} if out.exists() else {}

    def records(self, ps: list, asks: int = 1) -> list:
        """labels.jsonl of positions ps, each frame asked at most `asks` times; seeds by (2)."""
        recs = []
        for p in ps:
            answer, label = FRAMES[p][2][min(asks, len(FRAMES[p][2])) - 1]
            recs.append({"id": self.ids[p], "seed": SEED + p, "answer": answer})
            recs[-1]["label"] = label
        return recs

    def asked(self, first: int = 0) -> list:
        """Per ask_many call from `first` on: (frame positions of its prompts, its seeds)."""
        calls = self.fake.calls[first:]
        return [([self.fake.position(q) for q in c.prompts], c.seeds) for c in calls]
