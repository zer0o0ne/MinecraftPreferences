"""(1)-(3) of specs/clips.md: interval validity, state checks and the valid starts V."""

from collections.abc import Sequence

from mcpref.clips.evidence import Evidence
from mcpref.vpt.actions import interval_actions


def interval(
    ev: Evidence, kappa: Sequence[int], p: int, delta: int
) -> tuple[list[str], bool]:
    """Interface (1): the actions of interval p, a = kappa_p, b = kappa_{p+Delta}; valid
    iff n >= 1 and no line a..b-1 (cut lines included) is in F."""
    a, b = kappa[p], kappa[p + delta]
    acts = interval_actions(ev.segments, ev.lines, a, b, ev.cut, ev.turn_deg)
    return acts, len(acts) >= 1 and ev.foreign.isdisjoint(range(a, b))


def passes(ev: Evidence, t: int) -> bool:
    """(2): state t passes the line checks iff not Omega_t."""
    return not ev.modes[t]


def valid_starts(
    ev: Evidence, kappa: Sequence[int], delta: int, states: int
) -> list[int]:
    """Interface (3): the p whose clip exists, whose L - 1 intervals are valid by (1)
    and whose L states pass (2); ascending."""
    valid = [interval(ev, kappa, q, delta)[1] for q in range(len(kappa) - delta)]
    state_ok = [passes(ev, t) for t in kappa]
    return [
        p
        for p in range(len(kappa) - (states - 1) * delta)
        if all(valid[p + i * delta] for i in range(states - 1))
        and all(state_ok[p + i * delta] for i in range(states))
    ]
