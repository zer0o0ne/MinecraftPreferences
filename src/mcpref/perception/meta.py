"""(1), (2) of specs/perception.md: the frames v_s and meta texts mu_s of a clip; iota
of specs/vlm-labels.md (1)."""

from collections.abc import Mapping

import numpy as np

from mcpref.perception.config import PerceptionConfig
from mcpref.perception.cursor import Sprite, draw_cursor
from mcpref.pretrain.clip import Clip
from mcpref.vpt.yuv import yuv420p_to_rgb


def inventory_text(
    line: Mapping, inventory_line: str, item_line: str, no_items: str
) -> str:
    """(1) of specs/vlm-labels.md: iota, `inventory_line` with the line's inventory,
    quantities summed per type, types in ascending order."""
    if not line["inventory"]:
        return inventory_line.replace("{items}", no_items)
    totals = {}
    for entry in line["inventory"]:
        totals[entry["type"]] = totals.get(entry["type"], 0) + entry["quantity"]
    items = (
        item_line.replace("{type}", t).replace("{quantity}", str(totals[t]))
        for t in sorted(totals)
    )
    return inventory_line.replace("{items}", ", ".join(items))


def fmt(v: float, decimals: int) -> str:
    """(2): `decimals` places, the leading `-` removed when every digit is 0."""
    text = f"{v:.{decimals}f}"
    if text.startswith("-") and all(d == "0" for d in text if d.isdigit()):
        return text[1:]
    return text


def wrap(psi: float) -> float:
    """(2): yaw wrapped to [-180, 180)."""
    return ((psi + 180) % 360) - 180


def meta_text(cfg: PerceptionConfig, line: Mapping, first: Mapping) -> str:
    """(2): mu of `line`, x and z relative to the stream's first line `first`."""
    d = cfg.decimals
    values = {
        "{inventory}": inventory_text(
            line, cfg.inventory_line, cfg.item_line, cfg.no_items
        ),
        "{hotbar}": str(line["hotbar"] + 1),
        "{yaw}": fmt(wrap(line["yaw"]), d),
        "{pitch}": fmt(line["pitch"], d),
        "{x}": fmt(line["xpos"] - first["xpos"], d),
        "{y}": fmt(line["ypos"], d),
        "{z}": fmt(line["zpos"] - first["zpos"], d),
    }
    text = cfg.meta
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def clip_states(
    clip: Clip, cfg: PerceptionConfig, sprite: Sprite
) -> tuple[list[np.ndarray], list[str]]:
    """Notation: v_s (the cursor drawn by (1) where g_s) and mu_s (2) of each state."""
    states = clip.info["states"]
    lines = [clip.lines[t - states[0]] for t in states]
    frames, metas = [], []
    for s, line in enumerate(lines):
        v = yuv420p_to_rgb(clip.frames[s], clip.width, clip.height)
        if clip.info["gui"][s]:
            v = draw_cursor(v, line, sprite, cfg.mouse_height)
        frames.append(v)
        metas.append(meta_text(cfg, line, lines[0]))
    return frames, metas
