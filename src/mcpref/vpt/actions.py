"""Interface of specs/vpt-actions.md: config, (7)-(13); (1)-(6) in vpt/channels.py."""

import itertools
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from mcpref import number, positive_int
from mcpref.vpt.channels import (
    MOVES,
    Segment,
    grown,
    keys,
    passes,
    presses,
    segments,
    world,
)

__all__ = [
    "WORLD_ACTIONS",
    "ActionsConfig",
    "Segment",
    "foreign_lines",
    "interval_actions",
    "load_config",
    "mode_frames",
    "riding_lines",
    "segments",
    "starts_session",
]

WORLD_ACTIONS = (
    *MOVES.values(),
    *("sneak", "turn_left", "turn_right", "turn_up", "turn_down", "attack", "use"),
    *(f"hotbar_{i}" for i in range(1, 10)),
    "open_inventory",
)
RIDING = frozenset(
    f"minecraft.custom:minecraft.{m}_one_cm"
    for m in ("boat", "horse", "pig", "minecart", "strider")
)
C3 = frozenset({"a", "b", "c", "d", "f", "g", "h", "i", "n", "p", "q", "t", "f4"})


@dataclass(frozen=True)
class ActionsConfig:
    """Notation: configs/vpt-actions.yaml; turn_deg = rho, r_min = r_min."""

    turn_deg: float
    r_min: int


def load_config(path: Path) -> ActionsConfig:
    """Interface: the YAML file loaded into ActionsConfig; violated constraints raise."""
    cfg = ActionsConfig(**yaml.safe_load(path.read_text()))
    turn_deg = number("turn_deg", cfg.turn_deg)
    if not turn_deg > 0:
        raise ValueError(f"turn_deg {turn_deg!r} is not > 0")
    positive_int("r_min", cfg.r_min)
    return replace(cfg, turn_deg=turn_deg)


def wrap(x: float) -> float:
    """(7): x - 360 ceil((x - 180) / 360), in (-180, 180]."""
    return x - 360 * math.ceil((x - 180) / 360)


def turn_actions(angle: float, positive: str, negative: str, turn_deg: float) -> list:
    """(7): floor(|angle| / rho + 1/2) turns, `positive` if angle > 0, else `negative`."""
    n = math.floor(abs(angle) / turn_deg + 0.5)
    return [positive if angle > 0 else negative] * n


def turns(lines: Sequence[Mapping], a: int, b: int, turn_deg: float) -> list[str]:
    """(7): the n_psi yaw turns, then the n_theta pitch turns between the states a < b."""
    yaw = [float(lines[k]["yaw"]) for k in range(a, b + 1)]
    d_yaw = sum(wrap(y1 - y0) for y0, y1 in itertools.pairwise(yaw))
    d_pitch = float(lines[b]["pitch"]) - float(lines[a]["pitch"])
    yaw_turns = turn_actions(d_yaw, "turn_right", "turn_left", turn_deg)
    return yaw_turns + turn_actions(d_pitch, "turn_down", "turn_up", turn_deg)


def interval_actions(
    segs: Sequence[Segment],
    lines: Sequence[Mapping],
    a: int,
    b: int,
    cut: Collection[int],
    turn_deg: float,
) -> list[str]:
    """Interface (8): the actions of S_I in the order of segs, then the turns (7)."""
    inputs = set(range(a, b)).difference(cut)
    hit = [
        s.action
        for s in segs
        if not inputs.isdisjoint(range(max(s.first, a), min(s.last, b - 1) + 1))
    ]
    return hit + turns(lines, a, b, turn_deg)


def released(before: Mapping, line: Mapping) -> bool:
    """(9): F3 release, f3 in K_{k-1}, not in K_k, and G_{k-1}."""
    gone = "f3" in keys(before, "keys") and "f3" not in keys(line, "keys")
    return gone and passes(before)


def overlay_toggles(lines: Sequence[Mapping]) -> list[int]:
    """(9): frame k + 1 of every F3 release that ends no combo (flag eta)."""
    frames, combo = [], False
    for k, line in enumerate(lines):
        if world(line) and "f3" in keys(line, "keys") and keys(line, "newKeys") & C3:
            combo = True
        if k >= 1 and released(lines[k - 1], line):
            if combo:
                combo = False
            else:
                frames.append(k + 1)
    return frames


def counts(frames: list[int], n: int) -> list[int]:
    """(9): the number of events at frames <= t, t = 0..n-1."""
    return [sum(f <= t for f in frames) for t in range(n)]


def mode_frames(lines: Sequence[Mapping], start: tuple[bool, bool, int]) -> np.ndarray:
    """Interface (9): Omega_t = O_t or H_t or V_t != 0, t = 0..T-1."""
    o0, h0, v0 = start
    hud = [k + 2 for k in presses(lines, "f1")]
    view = [k + 1 for k in presses(lines, "f5")]
    n = len(lines)
    modes = zip(counts(overlay_toggles(lines), n), counts(hud, n), counts(view, n))
    omega = [
        (o0 + o) % 2 == 1 or (h0 + h) % 2 == 1 or (v0 + v) % 3 != 0 for o, h, v in modes
    ]
    return np.array(omega, dtype=bool)


def foreign_lines(lines: Sequence[Mapping]) -> np.ndarray:
    """Interface (10): W_k, f3 not in K_k and N_k meets {q, f}."""
    return np.array(
        [
            world(x)
            and "f3" not in keys(x, "keys")
            and bool(keys(x, "newKeys") & {"q", "f"})
            for x in lines
        ],
        dtype=bool,
    )


def riding_lines(lines: Sequence[Mapping]) -> np.ndarray:
    """Interface (11): a riding counter grows from l_k to l_{k+1}; the last line false."""
    rode = [bool(grown(x, after) & RIDING) for x, after in itertools.pairwise(lines)]
    return np.array([*rode, False], dtype=bool)


def starts_session(lines: Sequence[Mapping]) -> bool:
    """Interface (13): `stats` of l_0 is empty."""
    return not lines[0]["stats"]
