"""(1), (2) of specs/vlm-labels.md: the prompt of a pair of clips, the label of an
answer; Notation: the shown state of a clip, the requests of the comparisons."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from mcpref.perception.meta import inventory_text
from mcpref.vpt.yuv import yuv420p_to_rgb

if TYPE_CHECKING:
    from mcpref.preferences.labels import VlmLabelsConfig

INVALID, NO_ANSWER = "invalid", "no_answer"
MARKS = "`*\"'"
REMOVED = str.maketrans("", "", MARKS)


class Shown(NamedTuple):
    """Interface: the shown RGB frames of a clip, oldest first, and lambda, the line of
    its compared state."""

    frames: list[np.ndarray]
    line: Mapping


def compared_line(data: bytes) -> dict:
    """Notation: lambda_x, the last line of a lines.jsonl, parsed by json.loads."""
    return json.loads(data.removesuffix(b"\n").rsplit(b"\n", 1)[-1])


def read_shown(clip: Path, cfg: "VlmLabelsConfig", run: dict) -> Shown:
    """Notation: frames j_q = L - 1 - (k_V - 1 - q) s_V of the clip's frames.yuv as RGB,
    oldest first (W, H, L of Q_c's run.json); lambda_x."""
    W, H, L = (run["config"][k] for k in ("width", "height", "states"))
    beta = W * H * 3 // 2
    data = (clip / "frames.yuv").read_bytes()
    js = [L - 1 - (cfg.frames - 1 - q) * cfg.stride for q in range(cfg.frames)]
    frames = [yuv420p_to_rgb(data[j * beta : (j + 1) * beta], W, H) for j in js]
    return Shown(frames, compared_line((clip / "lines.jsonl").read_bytes()))


def requests(cfg: "VlmLabelsConfig", n: int, start: int) -> list[tuple[int, ...]]:
    """Notation: (c, o, i, seed) of the requests (o, i) of comparisons c = 0..n-1 in
    order, the first at position `start`: position start + 2sc + os + i."""
    s = cfg.samples
    return [
        (c, o, i, cfg.seed + start + 2 * s * c + o * s + i)
        for c in range(n)
        for o in (0, 1)
        for i in range(s)
    ]


def inventory(cfg: "VlmLabelsConfig", line: Mapping) -> str:
    """(1): iota(x) of lambda_x, by `inventory_text` of specs/perception.md."""
    return inventory_text(line, cfg.inventory_line, cfg.item_line, cfg.no_items)


def question(cfg: "VlmLabelsConfig") -> str:
    """(1): theta, `question` with {answers} replaced by one `answer_line` per entry of
    A in order."""
    lines = (
        cfg.answer_line.replace("{token}", e["token"]).replace(
            "{description}", e["description"]
        )
        for e in cfg.answers
    )
    return cfg.question.replace("{answers}", "\n".join(lines))


def prompt(
    cfg: "VlmLabelsConfig", first: Shown, second: Shown
) -> list[str | np.ndarray]:
    """(1): the parts of the prompt of (x, y) = (first, second)."""
    return [
        cfg.instruction,
        cfg.first,
        *first.frames,
        inventory(cfg, first.line),
        cfg.second,
        *second.frames,
        inventory(cfg, second.line),
        question(cfg),
    ]


def parse(answer: str | None, cfg: "VlmLabelsConfig") -> str:
    """(2): the label of an answer: a token of A, `invalid` or `no_answer`."""
    if answer is None:
        return NO_ANSWER
    pieces = [p.strip() for p in answer.translate(REMOVED).split("\n")]
    lines = [p for p in pieces if p]
    if not lines:
        return INVALID
    last = lines[-1].removesuffix(".").strip()
    tokens = {(cfg.prefix + e["token"]).strip(): e["token"] for e in cfg.answers}
    return tokens.get(last, INVALID)
