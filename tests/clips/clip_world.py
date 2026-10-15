"""Episodes for the tests of specs/clips.md: lines of tests/vpt/action_lines.py, hand-made
GuiActions, a recording `accept`, and seeded random episodes with cut and kept GUI spans."""

import numpy as np
from vpt.action_lines import CUSTOM, bump, episode

from mcpref.gui.actions import GuiActions
from mcpref.vpt.actions import ActionsConfig, Segment

ACTIONS = ActionsConfig(turn_deg=10.0, r_min=3)
OFF = (False, False, 0)
RIDE = CUSTOM + "boat_one_cm"
MOVES = ["-", "w", "w", "w+space", "space", "a", "s+d", "w+a", "left.shift", "w+M0",
         "M1"]  # fmt: skip
TAPS = {"q": 0.02, "f": 0.01, "f5": 0.002}
STATS = {CUSTOM + "jump": 0.05, "minecraft.mine_block:minecraft.stone": 0.03,
         "minecraft.use_item:minecraft.dirt": 0.03, RIDE: 0.015}  # fmt: skip
CLICKS = ["left_click_3", "right_click_7", "shift_click_12", "close"]


def gui(segs=(), cut=(), foreign=()) -> GuiActions:
    """Hand-made GuiActions of specs/vpt-gui-actions.md."""
    return GuiActions(
        segments=list(segs), cut=frozenset(cut), foreign=frozenset(foreign)
    )


class Log:
    """An `accept` recording its calls; rejects the clips whose first state is in `reject`."""

    def __init__(self, reject=()):
        self.calls, self.reject = [], set(reject)

    def __call__(self, states: list[int]) -> bool:
        self.calls.append(list(states))
        return states[0] not in self.reject


def world_tokens(rng, n: int, move: str) -> tuple[list[str], str]:
    """n world lines: the move combination changes w.p. 0.2 per line; taps of q and f; F1 or
    F3 tapped twice 2-6 lines apart (a mode shown for a few frames), F5 rarely."""
    keys = []
    for _ in range(n):
        if rng.random() < 0.2:
            move = MOVES[rng.integers(len(MOVES))]
        keys.append(
            ([] if move == "-" else move.split("+"))
            + [key for key, prob in TAPS.items() if rng.random() < prob]
        )
    for i in np.flatnonzero(rng.random(n) < 0.012):
        key, gap = ("f1", "f3")[rng.integers(2)], int(rng.integers(2, 7))
        for t in (i, i + gap)[: 1 + (i + gap < n)]:
            keys[t].append(key)
    return ["+".join(dict.fromkeys(k)) or "-" for k in keys], move


def span_actions(rng, k: int, j: int, container: bool) -> tuple[list, set]:
    """GUI segments and foreign lines of the kept span [k, j], none on the first line
    k >= 1 of a container."""
    segs, foreign = [], set()
    for t in range(k + (container and k >= 1), j + 1):
        if rng.random() < 0.3:
            segs.append(Segment(CLICKS[rng.integers(len(CLICKS))], t, t))
        if rng.random() < 0.1:
            segs.append(
                Segment("double_click_5", t, min(j, t + 1 + int(rng.integers(2))))
            )
        if rng.random() < 0.08:
            foreign.add(t)
    return segs, foreign


def random_tokens(rng, T: int) -> tuple[list[str], GuiActions]:
    """Alternating world runs of 1-24 lines and GUI spans of 1-8 lines, each cut w.p. 0.4."""
    tokens, segs, cut, foreign, move = [], [], set(), set(), "w"
    in_gui = rng.random() < 0.25
    while len(tokens) < T:
        if in_gui:
            k, n, kind = len(tokens), int(rng.integers(1, 9)), "#@"[rng.integers(2)]
            tokens += [f"{kind}:{move if rng.random() < 0.5 else '-'}"] * n
            j = min(len(tokens), T) - 1
            if rng.random() < 0.4:
                cut |= set(range(k, j + 1))
            else:
                s, f = span_actions(rng, k, j, kind == "#")
                segs, foreign = segs + s, foreign | f
        else:
            new, move = world_tokens(rng, int(rng.integers(1, 25)), move)
            tokens += new
        in_gui = not in_gui
    return tokens[:T], gui(segs, cut, foreign)


def random_episode(seed: int, T: int) -> tuple[list[dict], GuiActions, tuple]:
    """(lines, GuiActions, start): random tokens, hotbar, yaw (with whole-turn jumps), pitch,
    stats effects and riding."""
    rng = np.random.default_rng(seed)
    tokens, g = random_tokens(rng, T)
    changes = np.maximum.accumulate((rng.random(T) < 0.04) * np.arange(T))
    hotbar = (changes * 7 % 9).tolist()  # a slot set at each change line
    yaw = np.where(rng.random(T) < 0.4, rng.normal(0, 8, T), 0.0)
    yaw += 360.0 * (rng.random(T) < 0.01) * rng.integers(-8, 9, T)
    pitch = np.where(rng.random(T) < 0.3, rng.normal(0, 6, T), 0.0)
    pitch = np.clip(np.cumsum(pitch), -90, 90)
    lines = episode(" ".join(tokens), hotbar, np.cumsum(yaw).tolist(), pitch.tolist())
    for stat, prob in STATS.items():
        bump(lines, stat, *np.flatnonzero(rng.random(T - 1) < prob).tolist())
    return lines, g, (bool(rng.random() < 0.1), False, 0)
