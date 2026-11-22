"""Chronological observations with metadata-only lookahead and disk graph queries."""

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.datasets.continuous.views.reader import reader_for
from mcpref.datasets.vpt.video.yuv import yuv420p_to_rgb
from mcpref.gui.mask import load_config as mask_config
from mcpref.gui.mask import load_masks
from mcpref.models.perception.cursor import draw_cursor
from mcpref.models.perception.meta import meta_text
from mcpref.training.pretrain.continuous.data.config import IndexConfig, mathematics
from mcpref.training.pretrain.continuous.data.fingerprints import FORMAT, sources
from mcpref.training.pretrain.continuous.data.metadata import rows


class Index:
    def __init__(self, path):
        self.path = path
        self.run = json.loads((path / "run.json").read_text())
        if self.run.get("format") != FORMAT or not self.run.get("complete"):
            raise ValueError("A complete continuous training index is required")
        if digest(path / "catalogue.sqlite") != self.run["catalogue_sha256"]:
            raise ValueError("Continuous index catalogue checksum differs")
        self.db = sqlite3.connect(f"file:{path / 'catalogue.sqlite'}?mode=ro", uri=True)

    def close(self):
        self.db.close()

    def validate(self, config_path=None):
        config_path = config_path or REPO / self.run["index_config"]
        cfg = IndexConfig(**self.run["config"])
        maths = mathematics(cfg)
        masks = load_masks(mask_config(REPO / maths["action_mask"]))
        fingerprint, _ = sources(cfg, config_path, maths, masks)
        if fingerprint != self.run["fingerprint"]:
            raise ValueError("Continuous index inputs changed")

    def fragments(self, part):
        for (body,) in self.db.execute(
            "SELECT body FROM fragments WHERE part=? ORDER BY rowid", (part,)
        ):
            item = json.loads(body)
            if item["observations"]:
                yield item

    def endpoints(self, fragment_id, observation_id):
        query = "SELECT id,body,degree FROM endpoints WHERE fragment=? AND completion=? AND degree>0 ORDER BY id"
        for key, body, degree in self.db.execute(query, (fragment_id, observation_id)):
            originals = [
                row[0]
                for row in self.db.execute(
                    "SELECT original_id FROM aliases WHERE endpoint=? ORDER BY path",
                    (key,),
                )
            ]
            yield json.loads(body) | {
                "pending_edges": degree,
                "original_ids": originals,
            }

    def edges(self, endpoint_id):
        for (body,) in self.db.execute(
            "SELECT body FROM edges WHERE a=? OR b=? ORDER BY id",
            (endpoint_id, endpoint_id),
        ):
            yield json.loads(body)

    def excluded_pairs(self):
        for reason, body in self.db.execute(
            "SELECT reason,body FROM excluded ORDER BY id"
        ):
            yield json.loads(body) | {"reason": reason}


@dataclass(frozen=True)
class Observation:
    id: str
    index: int
    grid_index: int
    frame: np.ndarray
    line: dict
    origin: dict
    meta: str
    interval: dict
    mask: np.ndarray
    endpoints: tuple[dict, ...]
    shard: int
    position: int
    source: str


def outgoing(current, following):
    if following is not None and following["incoming"] is not None:
        return following["incoming"]
    return {
        "a": current["id"],
        "b": following["id"] if following else None,
        "actions": [],
        "legal_actions": [],
        "n": 0,
        "known": False,
        "valid": False,
    }


class StreamReader:
    def __init__(self, index, perception_cfg, sprite):
        self.index, self.cfg, self.sprite = index, perception_cfg, sprite
        self.root = REPO / index.run["config"]["dataset"]
        self.masks = load_masks(
            mask_config(REPO / index.run["mathematics"]["action_mask"])
        )
        self.payload = reader_for(self.root, index.run["width"], index.run["height"])

    def observations(self, fragment, start=0):
        if not 0 <= start <= fragment["observations"]:
            raise ValueError("Continuous stream cursor is outside the fragment")
        source = iter(rows(self.root, fragment, self.masks))
        current, following = next(source, None), next(source, None)
        while current is not None:
            if current["index"] >= start:
                raw = self.payload.frame(current["shard"], current["position"])
                frame = yuv420p_to_rgb(
                    raw, self.index.run["width"], self.index.run["height"]
                )
                if current["gui"]:
                    frame = draw_cursor(
                        frame, current["meta"], self.sprite, self.cfg.mouse_height
                    )
                interval = outgoing(current, following)
                yield Observation(
                    current["id"],
                    current["index"],
                    current["grid_index"],
                    frame,
                    current["meta"],
                    fragment["origin"],
                    meta_text(self.cfg, current["meta"], fragment["origin"]),
                    interval,
                    np.array(current["mask"], dtype=bool),
                    tuple(self.index.endpoints(fragment["id"], current["id"])),
                    current["shard"],
                    current["position"],
                    current["source"],
                )
            current, following = following, next(source, None)
