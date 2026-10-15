"""(1)-(6) of specs/vpt-actions.md: channel values, effects and the segments S."""

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

DIRECTIONS = {
    (1, 0): "forward",
    (-1, 0): "back",
    (0, -1): "left",
    (0, 1): "right",
    (1, -1): "forward_left",
    (1, 1): "forward_right",
    (-1, -1): "back_left",
    (-1, 1): "back_right",
}
MOVES = (
    {(f, r, 0): name for (f, r), name in DIRECTIONS.items()}
    | {(0, 0, 1): "jump"}
    | {(f, r, 1): f"jump_{name}" for (f, r), name in DIRECTIONS.items()}
)
JUMP = "minecraft.custom:minecraft.jump"
DAMAGE = frozenset(
    f"minecraft.custom:minecraft.damage_dealt{s}"
    for s in ("", "_absorbed", "_resisted")
)


@dataclass(frozen=True)
class Segment:
    """Notation: segment s = (action, first line, last line), inclusive."""

    action: str
    first: int
    last: int


def world(line: Mapping) -> bool:
    """Notation: W_k, isGuiOpen false."""
    return not line["isGuiOpen"]


def passes(line: Mapping) -> bool:
    """Notation: G_k, W_k or isGuiInventory."""
    return world(line) or line["isGuiInventory"]


def keys(line: Mapping, field: str) -> set[str]:
    """Notation: K_k (`keys`) or N_k (`newKeys`), names without key.keyboard."""
    return {k.removeprefix("key.keyboard.") for k in line["keyboard"][field]}


def move_input(line: Mapping) -> tuple[int, int, int]:
    """(1): (f_k, r_k, j_k)."""
    held = keys(line, "keys")
    return (
        ("w" in held) - ("s" in held),
        ("d" in held) - ("a" in held),
        int("space" in held),
    )


def move_value(line: Mapping) -> str | None:
    """(1) move: the name of (f_k, r_k, j_k); none for (0, 0, 0) or without W_k."""
    return MOVES.get(move_input(line)) if world(line) else None


def held_values(lines: Sequence[Mapping], on: Callable, name: str) -> list:
    """(1) sneak, attack, use: `name` on the lines with W_k and its input on, else none."""
    return [name if world(x) and on(x) else None for x in lines]


def grown(line: Mapping, after: Mapping) -> set[str]:
    """(2), (11): the x with sigma_{k+1}[x] > sigma_k[x], absent stats = 0."""
    before = line["stats"]
    return {x for x, c in after["stats"].items() if c > before.get(x, 0)}


def attack_effect(xs: set[str]) -> bool:
    """(2) attack: a mine_block or damage_dealt counter grows."""
    return any(x.startswith("minecraft.mine_block:") or x in DAMAGE for x in xs)


def use_effect(xs: set[str]) -> bool:
    """(2) use: a use_item counter grows."""
    return any(x.startswith("minecraft.use_item:") for x in xs)


def runs(values: Sequence) -> list[tuple[int, int, object]]:
    """Notation: the maximal runs [p, q] of equal values, with their value."""
    out, p = [], 0
    for v, group in itertools.groupby(values):
        n = len(list(group))
        out.append((p, p + n - 1, v))
        p += n
    return out


def between(values: Sequence, p: int, q: int) -> bool:
    """(3): lines p - 1 and q + 1 exist and have non-none values (hence W)."""
    inner = 0 < p and q + 1 < len(values)
    return inner and values[p - 1] is not None and values[q + 1] is not None


def fill_releases(values: list, worlds: list[bool], r_min: int) -> list:
    """(3a): short none runs of world lines between two equal values take that value."""
    out = list(values)
    for p, q, (v, w) in runs(list(zip(values, worlds))):
        short = v is None and w and q - p + 1 < r_min and between(values, p, q)
        if short and values[p - 1] == values[q + 1]:
            out[p : q + 1] = [values[p - 1]] * (q - p + 1)
    return out


def merge_short(values: list, jump: list[int], r_min: int) -> list:
    """(3b): in line order, short runs with j = 0 between values take the one before."""
    out = list(values)
    for p, q, v in runs(values):
        short = v is not None and q - p + 1 < r_min and not any(jump[p : q + 1])
        if short and between(out, p, q):
            out[p : q + 1] = [out[p - 1]] * (q - p + 1)
    return out


def smooth(lines: Sequence[Mapping], jump: list[int], r_min: int) -> list:
    """(3): the move values after (3a), then (3b)."""
    worlds = [world(line) for line in lines]
    moves = fill_releases([move_value(line) for line in lines], worlds, r_min)
    return merge_short(moves, jump, r_min)


def pieces(p: int, q: int, effects: Sequence[bool], r_min: int) -> list[tuple]:
    """(4): [p, q] cut after its split lines k; dropping short first and last pieces
    keeps the k with [p, k] and [k + 1, q] both at least r_min lines long."""
    cuts = [k for k in range(p, q) if effects[k] and min(k - p + 1, q - k) >= r_min]
    return [(s + 1, e) for s, e in itertools.pairwise([p - 1, *cuts, q])]


def run_segments(values: list, effects: Sequence[bool], r_min: int) -> list[Segment]:
    """(4): the segments of one channel."""
    return [
        Segment(v, first, last)
        for p, q, v in runs(values)
        if v is not None
        for first, last in pieces(p, q, effects, r_min)
    ]


def hotbar_segments(lines: Sequence[Mapping]) -> list[Segment]:
    """(5): delta = (h_{k+1} - h_k) mod 9 right steps if delta <= 4, else 9 - delta left."""
    segs = []
    for k, (line, after) in enumerate(itertools.pairwise(lines)):
        if world(line) and after["hotbar"] != line["hotbar"]:
            delta = (after["hotbar"] - line["hotbar"]) % 9
            if delta <= 4:
                segs += [Segment("hotbar_right", k, k)] * delta
            else:
                segs += [Segment("hotbar_left", k, k)] * (9 - delta)
    return segs


def inventory_segments(lines: Sequence[Mapping]) -> list[Segment]:
    """(6): (open_inventory, k, k) for W_k and e in N_k."""
    return [
        Segment("open_inventory", k, k)
        for k, line in enumerate(lines)
        if world(line) and "e" in keys(line, "newKeys")
    ]


def segments(lines: Sequence[Mapping], r_min: int) -> list[Segment]:
    """Interface: S by (1)-(6), by first line, then channel (8)."""
    grow = [grown(line, after) for line, after in itertools.pairwise(lines)]
    jump = [move_input(line)[2] for line in lines]
    jumped = [JUMP in g and j == 1 for g, j in zip(grow, jump)]
    sneak = held_values(lines, lambda x: "left.shift" in keys(x, "keys"), "sneak")
    attack = held_values(lines, lambda x: 0 in x["mouse"]["buttons"], "attack")
    use = held_values(lines, lambda x: 1 in x["mouse"]["buttons"], "use")
    channels = (
        (smooth(lines, jump, r_min), jumped),
        (sneak, [False] * len(grow)),
        (attack, [attack_effect(g) for g in grow]),
        (use, [use_effect(g) for g in grow]),
    )
    segs = [s for vals, effects in channels for s in run_segments(vals, effects, r_min)]
    segs += hotbar_segments(lines) + inventory_segments(lines)
    return sorted(segs, key=lambda s: s.first)
