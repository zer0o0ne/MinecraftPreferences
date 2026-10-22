"""Hand-made and random clips of specs/policy.md (Notation): `Clip`s of specs/pretrain.md whose
`info` holds the fields of specs/perception-data.md that the policy reads (`states`,
`actions`, `n`, `gui`, `classes`, `pitch`) and whose `lines` carry `hotbar`; the lines between
two states have hotbar 8 (so a mask read off the wrong line differs)."""

import numpy as np

from mcpref.pretrain.clip import Clip

FIRST, STEP = 100, 4  # the line of state 0, the lines from one state to the next
# tiny A for the network tests (|A| = 7)
NAMES = ("forward", "jump", "attack", "use", "turn_up", "hotbar_3", "left_click_1")
# names of the real A drawn by random clips: world actions, ones the world mask can forbid,
# GUI actions (forbidden at a world state)
POOL = ("forward", "jump", "attack", "use", "sneak", "turn_left", "turn_up", "turn_down",
        "hotbar_1", "hotbar_5", "hotbar_9", "close", "left_click_3", "shift_click_7")  # fmt: skip


def make_clip(actions: list[list[str]], pitch: list[float], hotbar: list[int],
              gui: list[bool] | None = None, classes: list | None = None,
              cid: str = "c") -> Clip:  # fmt: skip
    """`actions[s]`: the names of interval s; state s has pitch[s], its line hotbar[s];
    world states unless `gui`, `classes` given."""
    L = len(actions) + 1
    states = [FIRST + STEP * s for s in range(L)]
    lines = [{"hotbar": 8} for _ in range(states[-1] - FIRST + 1)]
    for s, h in zip(states, hotbar, strict=True):
        lines[s - FIRST]["hotbar"] = h
    info = {"id": cid, "states": states, "actions": actions,
            "n": [len(x) for x in actions], "gui": gui or [False] * L,
            "classes": classes or [None] * L, "pitch": pitch}  # fmt: skip
    return Clip(np.zeros((L, 6), dtype=np.uint8), 2, 2, info, lines)


def random_clip(rng: np.random.Generator, L: int, pool=POOL, cid: str = "r") -> Clip:
    """World states; n of an interval 1 (mostly), 2 or 3; pitch at and inside the limits."""
    n = rng.choice([1, 1, 1, 2, 3], L - 1)
    actions = [[str(a) for a in rng.choice(pool, k)] for k in n]
    pitch = rng.choice([-90.0, -30.0, 0.0, 45.0, 90.0], L).tolist()
    return make_clip(actions, pitch, rng.integers(0, 9, L).tolist(), cid=cid)


def random_clips(seed: int, m: int, L: int, pool=POOL) -> list[Clip]:
    rng = np.random.default_rng(seed)
    return [random_clip(rng, L, pool, f"r{i}") for i in range(m)]


# Hand clip (L = 8, intervals 0..6, targets of states 0..5), world states:
# state 4 has hotbar 2 (hotbar_3 masked there), state 5 pitch -90 (turn_up masked there).
HAND = make_clip(
    actions=[["forward"], ["jump"], ["attack", "attack", "turn_left"], ["use"],
             ["hotbar_3"], ["turn_up", "sneak"], ["left_click_3"]],
    pitch=[0.0, 0.0, 0.0, 0.0, 0.0, -90.0, 0.0, 0.0],
    hotbar=[0, 0, 0, 0, 2, 0, 0, 0],
    cid="hand",
)  # fmt: skip
