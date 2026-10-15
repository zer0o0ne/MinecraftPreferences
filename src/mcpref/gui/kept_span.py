"""(3)-(7) of specs/vpt-gui-actions.md: the segments and foreign lines of a kept span."""

import itertools
from collections.abc import Mapping, Sequence

from mcpref.gui.screens import Element, ScreenClass, element_at
from mcpref.vpt.actions import Segment
from mcpref.vpt.channels import keys

DIGITS = {f"key.keyboard.{s}": s for s in range(1, 10)}
SHIFT = frozenset({"left.shift", "right.shift"})


def pressed(line: Mapping) -> list[int]:
    """Notation: N^B_t meets {0, 1}, in the order of N^B_t."""
    return [b for b in line["mouse"]["newButtons"] if b in (0, 1)]


def new_digits(line: Mapping) -> list[int]:
    """Notation: the digits s in N_t, in the order of N_t."""
    return [DIGITS[k] for k in line["keyboard"]["newKeys"] if k in DIGITS]


def at(cls: ScreenClass | None, line: Mapping) -> Element | None:
    """Notation: e_t, element_at of `cls` at (x_t / 2, y_t / 2); none without a class."""
    if cls is None:
        return None
    return element_at(cls, line["mouse"]["x"] / 2, line["mouse"]["y"] / 2)


def shift(line: Mapping) -> bool:
    """Notation: shift_t, left.shift or right.shift in K_t."""
    return bool(keys(line, "keys") & SHIFT)


def is_slot(e: Element | None) -> bool:
    """(3a), (6): e is a slot."""
    return e is not None and e.kind == "slot"


def pairs(
    lines: Sequence[Mapping], left: list[int], e: Mapping, double_click_ms: int
) -> dict[int, int]:
    """(3a): first press t -> second press t' of each pair, consecutive left presses."""
    out = {}
    for t, t2 in itertools.pairwise(left):
        quick = lines[t2]["milli"] - lines[t]["milli"] < double_click_ms
        if t not in out.values() and is_slot(e[t]) and e[t2] == e[t] and quick:
            out[t] = t2
    return out


def click(line: Mapping, t: int, b: int, e: Element) -> Segment:
    """(3): shift_click with shift held, else left_click (b = 0) or right_click (b = 1)."""
    kind = "shift_click" if shift(line) else ("left_click", "right_click")[b]
    return Segment(f"{kind}_{e.index}", t, t)


def press_segments(
    lines: Sequence[Mapping], presses: list, e: Mapping, pair: Mapping[int, int]
) -> tuple[list[Segment], set[int]]:
    """(3), (3a): clicks, double clicks at their first press, and foreign lines."""
    segs, foreign = [], set()
    for t, b in presses:
        if b == 0 and t in pair:
            if shift(lines[pair[t]]):
                foreign.update(range(t, pair[t] + 1))
            else:
                segs.append(Segment(f"double_click_{e[t].index}", t, pair[t]))
        elif b == 0 and t in pair.values():
            continue
        elif e[t] is None:
            foreign.add(t)
        else:
            segs.append(click(lines[t], t, b, e[t]))
    return segs, foreign


def hold(lines: Sequence[Mapping], t: int, b: int, j: int) -> range:
    """(4): lines t..t' with b in B, within the span, before the next press of b."""
    r = t
    while (
        r < j
        and b in lines[r + 1]["mouse"]["buttons"]
        and b not in pressed(lines[r + 1])
    ):
        r += 1
    return range(t, r + 1)


def drag_lines(
    lines: Sequence[Mapping], presses: list, cls: Mapping, e: Mapping, j: int
) -> set[int]:
    """(4): the holds of presses with e_t not none whose e_r (class c_t) differs."""
    foreign = set()
    for t, b in presses:
        if e[t] is not None:
            held = hold(lines, t, b, j)
            if any(at(cls[t], lines[r]) != e[t] for r in held):
                foreign.update(held)
    return foreign


def close(lines: Sequence[Mapping], g: int, j: int) -> list[Segment]:
    """(5): (close, j, j) for j <= T - 2 and e or escape in N_j."""
    shut = keys(lines[j], "newKeys") & {"e", "escape"}
    return [Segment("close", j, j)] if g <= j <= len(lines) - 2 and shut else []


def swaps(cls: ScreenClass | None, e: Element | None, s: int) -> bool:
    """(6): e_t is a slot, iota(e_t) != s, and c_t has an element with index s."""
    return is_slot(e) and e.index != s and any(x.index == s for x in cls.elements)


def digit_segments(
    lines: Sequence[Mapping], t: int, cls: ScreenClass | None, e: Element | None
) -> list[Segment]:
    """(6): three left clicks per new digit s (not an auto-repeat) that swaps."""
    segs = []
    for s in new_digits(lines[t]):
        fresh = t == 0 or str(s) not in keys(lines[t - 1], "keys")
        if fresh and swaps(cls, e, s):
            segs += [Segment(f"left_click_{i}", t, t) for i in (e.index, s, e.index)]
    return segs


def key_lines(lines: Sequence[Mapping], g: int, j: int) -> set[int]:
    """(7): the lines with q or f in N_t."""
    return {t for t in range(g, j + 1) if keys(lines[t], "newKeys") & {"q", "f"}}


def kept_span(
    lines: Sequence[Mapping], cls: Mapping, double_click_ms: int, g: int, j: int
) -> tuple[list[Segment], set[int]]:
    """(3)-(7) on the lines g..j of a kept span, segments in the order of (9);
    `cls` the classes of its lines with a press or digit."""
    e = {t: at(c, lines[t]) for t, c in cls.items()}
    presses = [(t, b) for t in e for b in pressed(lines[t])]
    pair = pairs(lines, [t for t, b in presses if b == 0], e, double_click_ms)
    clicks, wrong = press_segments(lines, presses, e, pair)
    digits = [seg for t in e for seg in digit_segments(lines, t, cls[t], e[t])]
    segs = sorted(clicks + close(lines, g, j) + digits, key=lambda seg: seg.first)
    drags = drag_lines(lines, presses, cls, e, j)
    return segs, wrong | drags | key_lines(lines, g, j)
