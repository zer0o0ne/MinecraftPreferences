"""The inputs of `main` of specs/perception-data.md (revision of 2026-10-20): a fake gui-masks
output (classifier_world.py), configs/gui-classifier.yaml, the block's config with every path
relative to the repository root, and a fake `collect`."""

import sys

import numpy as np
import yaml
from classifier_world import Masks, rel, write_classifier_config

from mcpref.clips import run

from .collect_site import FIELDS

W, H = 6, 4
THRESHOLDS = {"agreement": (0, "0.9", "0.1"), "macro_recall": (0, "0.95", "0.05")}
PREFIX = "mcpref.clips.run: "
MARK = "fake collect called"


class FakeCollect:
    """`collect(cfg, classify, classifier_id)`: records cfg, the labels of `frames` and the
    id, marks stderr, then raises `error` if given."""

    def __init__(self, frames: list, error: BaseException | None = None):
        self.frames, self.error, self.calls = frames, error, []

    def __call__(self, cfg, classify, classifier_id) -> None:
        print(MARK, file=sys.stderr)
        self.calls.append((cfg, classify(self.frames), classifier_id))
        if self.error is not None:
            raise self.error


class World:
    """A fake gui-masks output with its catalogue, configs/gui-classifier.yaml and the
    block's config, every path relative to the repository root."""

    def __init__(self, tmp, repo, criterion: str = "macro_recall"):
        rng = np.random.default_rng(4)
        pair = lambda: (rng.integers(0, 256, (H, W, 3), dtype=np.uint8), rng.random((H, W)) < 0.6)  # fmt: skip
        classes = {"inv": {"m1": pair(), "m2": None}, "chest": {"main": pair()}}
        classes["empty"] = {"x": None}
        self.masks = Masks(tmp / "gm", repo, classes, THRESHOLDS, W, H).write()
        self.criterion, self.tmp, self.repo = criterion, tmp, repo
        self.classifier = write_classifier_config(
            tmp / "gc.yaml", self.masks, criterion
        )
        # the templates of inv and chest, then a random frame: inv, chest, other
        self.frames = [classes["inv"]["m1"][0], classes["chest"]["main"][0]]
        self.frames.append(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
        self.path = self.config()

    def config(self, **changes) -> str:
        """The block's config file; its path (absolute)."""
        doc = FIELDS | {
            "index_urls": {"7xx": "http://127.0.0.1:9/i7"},
            "ffmpeg": "/x/ffmpeg",
        }
        doc |= {"clips": "configs/clips.yaml", "screens": self.masks.screens}
        doc |= {"gui_actions": "configs/vpt-gui-actions.yaml"}
        doc |= {"overlay": "configs/f3-overlay.yaml"}
        doc |= {"classifier_config": rel(self.classifier, self.repo)}
        doc |= {"out_dir": rel(self.tmp / "out", self.repo), "width": W, "height": H}
        path = self.tmp / "perception-data.yaml"
        path.write_text(yaml.safe_dump(doc | changes, sort_keys=False))
        return str(path)


def patch_collect(monkeypatch, fake: FakeCollect) -> None:
    """The fake replaces `collect` wherever main may look it up."""
    monkeypatch.setattr("mcpref.clips.collect.collect", fake)
    monkeypatch.setattr(run, "collect", fake, raising=False)
