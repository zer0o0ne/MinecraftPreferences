"""Naive references of specs/clips.md: Notation (kappa, S, F, Omega, gui_t) and (1)-(5),
composed of the public functions of specs/vpt-actions.md exactly as the spec composes them."""

from dataclasses import dataclass

import numpy as np

from mcpref.vpt.actions import (
    Segment,
    foreign_lines,
    interval_actions,
    mode_frames,
    riding_lines,
    segments,
)


@dataclass(frozen=True)
class Truth:
    """The evidence of one episode by Notation, with kappa."""

    lines: list
    segs: list
    cut: frozenset
    foreign: frozenset
    modes: np.ndarray
    turn_deg: float
    kappa: list


def truth(lines, gui, start, actions) -> Truth:
    """Notation: S from the kept lines mapped back to kappa, merged by first line; F; Omega."""
    kappa = [t for t in range(len(lines)) if t not in gui.cut]
    kept_lines = [lines[k] for k in kappa]
    world = [
        Segment(s.action, kappa[s.first], kappa[s.last])
        for s in segments(kept_lines, actions.r_min)
    ]
    segs = sorted(world + list(gui.segments), key=lambda s: s.first)
    flags = foreign_lines(lines) | riding_lines(lines)
    foreign = frozenset(int(k) for k in np.flatnonzero(flags)) | gui.foreign
    modes = mode_frames(lines, start)
    return Truth(lines, segs, gui.cut, foreign, modes, actions.turn_deg, kappa)


def interval_truth(tr: Truth, p: int, delta: int) -> tuple[list[str], bool]:
    """(1): the actions of interval p and n >= 1 with no line of F in a..b-1."""
    a, b = tr.kappa[p], tr.kappa[p + delta]
    acts = interval_actions(tr.segs, tr.lines, a, b, tr.cut, tr.turn_deg)
    return acts, len(acts) >= 1 and all(k not in tr.foreign for k in range(a, b))


def starts_truth(tr: Truth, delta: int, states: int) -> list[int]:
    """(3) by trying every p."""
    found = []
    for p in range(len(tr.kappa)):
        if p + (states - 1) * delta > len(tr.kappa) - 1:
            continue
        steps = range(states - 1)
        intervals = all(interval_truth(tr, p + i * delta, delta)[1] for i in steps)
        lines = [tr.kappa[p + i * delta] for i in range(states)]
        if intervals and not any(tr.modes[t] for t in lines):
            found.append(p)
    return found


def gui_flag(lines, t: int) -> bool:
    """gui_t: `isGuiOpen`, false on the first line t >= 1 of a container span."""
    first = t >= 1 and not lines[t - 1]["isGuiOpen"]
    return lines[t]["isGuiOpen"] and not (first and not lines[t]["isGuiInventory"])


def clip_truth(tr: Truth, p: int, delta: int, states: int) -> dict:
    """(5): the fields of the clip at p."""
    lines = [tr.kappa[p + i * delta] for i in range(states)]
    acts = [interval_truth(tr, p + i * delta, delta)[0] for i in range(states - 1)]
    return {"p": p, "states": lines, "actions": acts, "n": [len(a) for a in acts],
            "gui": [gui_flag(tr.lines, t) for t in lines],
            "pitch": [tr.lines[t]["pitch"] for t in lines]}  # fmt: skip


def draws_truth(tr: Truth, starts, delta, states, m, g, accept) -> list[int]:
    """(4): the starts taken, in the order taken; spans compared as line ranges."""
    w, taken = list(starts), []
    while len(taken) < m and w:
        p = w[g.integers(len(w))]
        if accept([tr.kappa[p + i * delta] for i in range(states)]):
            taken.append(p)
            lo, hi = tr.kappa[p], tr.kappa[p + (states - 1) * delta]
            w = [
                q
                for q in w
                if hi < tr.kappa[q] or tr.kappa[q + (states - 1) * delta] < lo
            ]
        else:
            w.remove(p)
    return taken
