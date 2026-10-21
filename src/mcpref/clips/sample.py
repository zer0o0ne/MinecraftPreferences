"""(4), (5) of specs/clips.md: sequential sampling of non-overlapping clips."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mcpref.clips.evidence import Evidence
from mcpref.clips.validity import interval
from mcpref.gui.actions import opens_container


@dataclass(frozen=True)
class Clip:
    """(5): start p; per interval its actions and n; per state its line, gui_t, pitch."""

    p: int
    states: list[int]
    actions: list[list[str]]
    n: list[int]
    gui: list[bool]
    pitch: list[float]


def state_lines(kappa: Sequence[int], p: int, delta: int, states: int) -> list[int]:
    """Notation: the states kappa_{p + i Delta}, i = 0..L-1, of the clip at p."""
    return [kappa[p + i * delta] for i in range(states)]


def gui_flag(lines: Sequence[Mapping], t: int) -> bool:
    """Notation: gui_t, isGuiOpen of l_t, false on the first line of a container span."""
    return lines[t]["isGuiOpen"] and not opens_container(lines, t)


def clip(ev: Evidence, kappa: Sequence[int], p: int, delta: int, states: int) -> Clip:
    """(5): per interval p + i Delta its actions and n; per state gui_t and pitch."""
    acts = [interval(ev, kappa, p + i * delta, delta)[0] for i in range(states - 1)]
    ts = state_lines(kappa, p, delta, states)
    gui = [gui_flag(ev.lines, t) for t in ts]
    pitch = [float(ev.lines[t]["pitch"]) for t in ts]
    return Clip(p, ts, acts, [len(a) for a in acts], gui, pitch)


def uniform(g: np.random.Generator, n: int) -> int:
    """g.integers(n), its defaults low=0, dtype=np.int64, endpoint=False passed."""
    return int(g.integers(0, n, dtype=np.int64, endpoint=False))


def overlaps(kappa: Sequence[int], p: int, q: int, span: int) -> bool:
    """(4): the spans kappa_p..kappa_{p+span} and kappa_q..kappa_{q+span} share a line."""
    return kappa[q] <= kappa[p + span] and kappa[p] <= kappa[q + span]


def sample(
    ev: Evidence,
    kappa: Sequence[int],
    starts: Sequence[int],
    delta: int,
    states: int,
    m: int,
    g: np.random.Generator,
    accept: Callable[[list[int]], bool],
) -> list[Clip]:
    """Interface (4): uniform draws from W = V; a clip accepted is taken and the starts
    overlapping it leave W, a clip rejected leaves alone; clips in the order taken."""
    w, taken, span = list(starts), [], (states - 1) * delta
    while len(taken) < m and w:
        j = uniform(g, len(w))
        p = w[j]
        if accept(state_lines(kappa, p, delta, states)):
            taken.append(clip(ev, kappa, p, delta, states))
            w = [q for q in w if not overlaps(kappa, p, q, span)]
        else:
            del w[j]
    return taken
