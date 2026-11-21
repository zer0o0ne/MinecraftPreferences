"""Disk-backed fragment rows, with one bounded metadata page in memory."""

import gzip
import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from mcpref.datasets.vpt.actions import RIDING
from mcpref.datasets.vpt.video.channels import DAMAGE, JUMP

FIELDS = (
    "isGuiOpen",
    "isGuiInventory",
    "keyboard",
    "mouse",
    "milli",
    "serverTick",
    "hotbar",
    "yaw",
    "pitch",
    "inventory",
    "xpos",
    "ypos",
    "zpos",
)


def projection(row: dict, index: int) -> dict:
    if type(row["tick"]) is not int or row["tick"] != index:
        raise ValueError(f"Unexpected source tick at row {index}")
    result = {key: row[key] for key in FIELDS}
    for key in ("milli", "serverTick", "hotbar"):
        if type(result[key]) is not int:
            raise ValueError(f"Noninteger {key}")
    if not 0 <= result["hotbar"] < 9:
        raise ValueError("Invalid hotbar slot")
    for key in ("yaw", "pitch", "xpos", "ypos", "zpos"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
            raise ValueError(f"Nonfinite {key}")
    for key in ("isGuiOpen", "isGuiInventory"):
        if type(result[key]) is not bool:
            raise ValueError(f"Nonboolean {key}")
    for key in ("keys", "newKeys"):
        if not isinstance(result["keyboard"][key], list) or any(
            type(v) is not str for v in result["keyboard"][key]
        ):
            raise TypeError(f"Invalid keyboard.{key}")
    for key in ("buttons", "newButtons"):
        if not isinstance(result["mouse"][key], list) or any(
            type(v) is not int for v in result["mouse"][key]
        ):
            raise TypeError(f"Invalid mouse.{key}")
    for key in ("x", "y"):
        if type(result["mouse"][key]) not in (int, float) or not math.isfinite(
            result["mouse"][key]
        ):
            raise TypeError(f"Invalid mouse.{key}")
    for item in result["inventory"]:
        if (
            not isinstance(item["type"], str)
            or type(item["quantity"]) not in (int, float)
            or not math.isfinite(item["quantity"])
        ):
            raise TypeError("Invalid inventory")
    result["stats"] = {
        k: v
        for k, v in row["stats"].items()
        if k in RIDING
        or k in DAMAGE
        or k == JUMP
        or k.startswith(("minecraft.mine_block:", "minecraft.use_item:"))
    }
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for v in result["stats"].values()
    ):
        raise ValueError("Invalid action counter")
    result["starts_session"] = not row["stats"]
    return result


class Rows(Sequence):
    def __init__(self, path: Path, page_size: int):
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS rows (i INTEGER PRIMARY KEY, body TEXT)"
        )
        self.size = self.db.execute("SELECT count(*) FROM rows").fetchone()[0]
        self.page_size, self.page, self.values = page_size, -1, []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        page = index // self.page_size
        if page != self.page:
            first = page * self.page_size
            cursor = self.db.execute(
                "SELECT body FROM rows WHERE i >= ? AND i < ? ORDER BY i",
                (first, first + self.page_size),
            )
            self.values = [json.loads(body) for (body,) in cursor]
            self.page = page
        return self.values[index % self.page_size]

    def append_source(self, path: Path, piece: dict):
        """Rollback the entire source on malformed rows; preserve original gzip separately."""
        first_index, first, previous = self.size, None, None
        try:
            with gzip.open(path, "rt") as stream:
                for i, text in enumerate(stream):
                    row = projection(json.loads(text), i)
                    if previous and (
                        row["milli"] < previous["milli"]
                        or row["serverTick"] < previous["serverTick"]
                    ):
                        raise ValueError("Source clocks decrease inside a piece")
                    self.db.execute(
                        "INSERT INTO rows VALUES (?, ?)",
                        (self.size, json.dumps(row, separators=(",", ":"))),
                    )
                    self.size += 1
                    first, previous = row if first is None else first, row
            boundary = piece["observed_boundary"]
            if self.size - first_index != piece["nominal_steps"]:
                raise ValueError("Source row count differs from manifest")
            if (
                "first_stats_empty" in boundary
                and first["starts_session"] != boundary["first_stats_empty"]
            ):
                raise ValueError("Source world-start marker differs from manifest")
            if (
                first["milli"] != boundary["first_milli"]
                or previous["milli"] != boundary["last_milli"]
            ):
                raise ValueError("Source boundary differs from manifest")
            if (
                first["serverTick"] != boundary["first_server_tick"]
                or previous["serverTick"] != boundary["last_server_tick"]
            ):
                raise ValueError("Source server ticks differ from manifest")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.size = first_index
            raise
        self.page = -1
        return first_index

    def close(self):
        self.db.close()
