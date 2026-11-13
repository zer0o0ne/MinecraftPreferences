"""Recognise recorded RL cycles without inventing a global optimizer step."""

import re
from pathlib import Path

from mcpref.config import REPO


def cycle_view(entry: dict) -> tuple | None:
    if "identity" in entry:
        kind, producer, source, _split, _condition = entry["identity"]
        match = re.fullmatch(r"(ppo|shift|update|memory):(\d+)", producer)
        if kind != "rl" or match is None:
            return None
        role, cycle = match[1], int(match[2])
        path = REPO / source
        if role == "memory":
            path = path.parent
        phase = "joint"
    else:
        match = re.match(r"RL cycle (\d+) / (joint|retrain)", entry["population"])
        if match is None:
            return None
        cycle, phase = int(match[1]), match[2]
        path = Path(entry["source"])
        if phase == "retrain":
            path = path.parent
        role = "saved"
    if not path.name.isdigit() or int(path.name) != cycle:
        return None
    return str(path.parent.resolve()), role, phase, cycle
