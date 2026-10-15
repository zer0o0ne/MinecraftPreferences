"""Random episodes against the naive transcription of specs/vpt-actions.md
(actions_reference.py): segments (1)-(6), (8); interval_actions (7), (8); modes (9); (10);
(11)."""

import numpy as np
import pytest
from action_lines import CUSTOM, line
from actions_reference import foreign, interval, modes, riding
from actions_reference import segments as reference_segments

from mcpref.vpt.actions import (
    Segment,
    foreign_lines,
    interval_actions,
    mode_frames,
    riding_lines,
    segments,
)

MOVE_KEYS = dict.fromkeys(
    ["w", "a", "s", "d", "space", "left.shift", "left.control", "e", "q", "f"], 0.12
)
MODE_KEYS = {  # one mode at a time (the others off at the start), then all: Omega is their union
    "overlay": dict.fromkeys(["f3", "a", "b", "f4", "q", "f", "e", "w", "s"], 0.12),
    "hud": {"f1": 0.12, "w": 0.12},
    "view": {"f5": 0.12, "e": 0.12},
    "all": dict.fromkeys(["f3", "a", "f4", "w", "s"], 0.12) | {"f1": 0.03, "f5": 0.03},
}
STARTS = {"overlay": (1, 0, 1), "hud": (0, 1, 1), "view": (0, 0, 3), "all": (1, 1, 3)}
STATS = [CUSTOM + "jump", "minecraft.mine_block:minecraft.stone",
         "minecraft.mine_block:minecraft.dirt", CUSTOM + "damage_dealt",
         CUSTOM + "damage_dealt_absorbed", CUSTOM + "damage_dealt_resisted",
         "minecraft.use_item:minecraft.bread", "minecraft.use_item:minecraft.oak_planks",
         CUSTOM + "damage_taken", CUSTOM + "boat_one_cm", CUSTOM + "pig_one_cm",
         CUSTOM + "walk_one_cm"]  # fmt: skip
N_EPISODES, T = 300, 60


def random_episode(
    seed: int, keys: dict[str, float], p_gui: float = 0.06
) -> list[dict]:
    """Sticky keys (flipped with their probability) and buttons with auto-repeats, GUI spans,
    hotbar, quarter-degree turns, stats growing at random."""
    rng = np.random.default_rng(seed)
    held, buttons, gui, hotbar, yaw, pitch, stats = set(), set(), "", 0, 0.0, 0.0, {}
    lines = []
    for k in range(T):
        gui = str(rng.choice(["", "", "#", "@"])) if rng.random() < p_gui else gui
        now = held ^ {x for x, p in keys.items() if rng.random() < p}
        new = (now - held) | {x for x in now & held if rng.random() < 0.03}
        buttons ^= {b for b in (0, 1, 2) if rng.random() < 0.12}
        hotbar = int(rng.integers(9)) if rng.random() < 0.08 else hotbar
        jump = 360 * int(rng.integers(-8, 9)) if rng.random() < 0.03 else 0
        yaw += 0.25 * int(rng.integers(-800, 801)) * (rng.random() < 0.5) + jump
        pitch = 0.25 * int(rng.integers(-360, 361)) if rng.random() < 0.3 else pitch
        stats.update({x: stats.get(x, 0) + 1 for x in STATS if rng.random() < 0.06})
        ln = line(sorted(now), sorted(new), sorted(buttons), gui, k, hotbar, yaw, pitch)
        ln["stats"].update(stats)
        lines.append(ln)
        held = now
    return lines


def as_tuples(segs) -> list:
    return [(s.action, s.first, s.last) for s in segs]


@pytest.mark.parametrize("r_min", [1, 2, 3, 4, 6])
def test_segments_match_reference(r_min):
    for seed in range(N_EPISODES):
        lines = random_episode(seed, MOVE_KEYS)
        found = segments(lines, r_min=r_min)
        assert all(type(s) is Segment for s in found)
        assert as_tuples(found) == reference_segments(lines, r_min), f"seed {seed}"


@pytest.mark.parametrize("turn_deg", [10.0, 2.5, 7.25])
def test_interval_actions_match_reference(turn_deg):
    rng = np.random.default_rng(1)
    for seed in range(N_EPISODES):
        lines = random_episode(seed, MOVE_KEYS)
        segs = segments(lines, r_min=3)
        a = int(rng.integers(T - 1))
        b = int(rng.integers(a + 1, min(a + 13, T)))
        cut = {k for k in range(a - 2, b + 2) if rng.random() < 0.2}
        expected = interval(as_tuples(segs), lines, a, b, cut, turn_deg)
        assert interval_actions(segs, lines, a, b, cut, turn_deg) == expected, (
            f"seed {seed}"
        )


@pytest.mark.parametrize("family", list(MODE_KEYS))
def test_mode_frames_match_reference(family):
    rng, (o, h, v) = np.random.default_rng(2), STARTS[family]
    for seed in range(N_EPISODES):
        lines = random_episode(seed, MODE_KEYS[family], p_gui=0.2)
        start = (
            bool(rng.random() < 0.5 * o),
            bool(rng.random() < 0.5 * h),
            int(rng.integers(v)),
        )
        found = mode_frames(lines, start)
        assert found.tolist() == modes(lines, start), f"seed {seed}"


def test_foreign_and_riding_lines_match_reference():
    for seed in range(N_EPISODES):
        lines = random_episode(seed, MODE_KEYS["overlay"])
        assert foreign_lines(lines).tolist() == foreign(lines), f"seed {seed}"
        assert riding_lines(lines).tolist() == riding(lines), f"seed {seed}"
