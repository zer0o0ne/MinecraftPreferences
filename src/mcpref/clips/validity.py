"""(1)-(3) of specs/clips.md: interval validity, state checks and the valid starts V."""

from collections.abc import Sequence
from dataclasses import dataclass

from mcpref.clips.evidence import Evidence
from mcpref.vpt.actions import interval_actions, intervals_actions


@dataclass(frozen=True)
class IntervalCache:
    """Actions and validity of consecutive intervals on one kept-line grid."""

    actions: list[list[str]]
    valid: list[bool]


def interval(
    ev: Evidence, kappa: Sequence[int], p: int, delta: int
) -> tuple[list[str], bool]:
    """Interface (1): the actions of interval p, a = kappa_p, b = kappa_{p+Delta}; valid
    iff n >= 1 and no line a..b-1 (cut lines included) is in F."""
    a, b = kappa[p], kappa[p + delta]
    acts = interval_actions(ev.segments, ev.lines, a, b, ev.cut, ev.turn_deg)
    return acts, len(acts) >= 1 and ev.foreign.isdisjoint(range(a, b))


def cache_intervals(ev: Evidence, kappa: Sequence[int], delta: int) -> IntervalCache:
    """The consecutive interval results of (1), with one ordered segment sweep."""
    actions = intervals_actions(ev.segments, ev.lines, kappa, delta, ev.turn_deg)
    foreign = [0]
    for t in range(len(ev.lines)):
        foreign.append(foreign[-1] + (t in ev.foreign))
    valid = []
    for p, acts in enumerate(actions):
        a, b = kappa[p], kappa[p + delta]
        valid.append(bool(acts) and foreign[b] - foreign[a] == 0)
    return IntervalCache(actions, valid)


def passes(ev: Evidence, t: int) -> bool:
    """(2): state t passes the line checks iff not Omega_t."""
    return not ev.modes[t]


def _window_bad(values: Sequence[bool], step: int, size: int, starts: int) -> list[int]:
    """Numbers of true values in each length-size window of the given arithmetic grids."""
    counts = [0] * starts
    for residue in range(step):
        prefix = [0]
        for q in range(residue, len(values), step):
            prefix.append(prefix[-1] + values[q])
        for j in range(len(prefix) - size):
            p = residue + j * step
            if p >= starts:
                break
            counts[p] = prefix[j + size] - prefix[j]
    return counts


def valid_starts(
    ev: Evidence,
    kappa: Sequence[int],
    delta: int,
    states: int,
    *,
    cache: IntervalCache | None = None,
) -> list[int]:
    """Interface (3): the p whose clip exists, whose L - 1 intervals are valid by (1)
    and whose L states pass (2); ascending."""
    cache = cache_intervals(ev, kappa, delta) if cache is None else cache
    n = max(0, len(kappa) - (states - 1) * delta)
    intervals = _window_bad([not ok for ok in cache.valid], delta, states - 1, n)
    state_bad = [not passes(ev, t) for t in kappa]
    modes = _window_bad(state_bad, delta, states, n)
    return [p for p in range(n) if intervals[p] == 0 and modes[p] == 0]
