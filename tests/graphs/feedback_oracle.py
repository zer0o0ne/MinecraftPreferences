"""Transcription of (2), (3) of specs/acyclic.md for its tests, written from the spec alone:
Eades' steps (i)-(iv) of its [I] in exact arithmetic with the lowest-id tie-break (equal to
igraph's float64 semantics on dyadic costs), the igraph call of (2) itself (the definition,
for costs where float64 rounding can break ties), the put-back (3) and the properties (3)
states."""

import random
from collections.abc import Callable, Sequence
from fractions import Fraction

import igraph

Arcs = Sequence[tuple[int, int]]


def live_arcs(arcs: Arcs, alive: set, v: int, end: int) -> list[int]:
    """The arc ids of v's out-arcs (end 0: v is the tail) or in-arcs (end 1) whose other end
    is alive."""
    return [e for e, a in enumerate(arcs) if a[end] == v and a[1 - end] in alive]


def strip(arcs: Arcs, alive: set) -> None:
    """(i): sources and sinks (no in-arc or no out-arc among the alive, counted by arcs)
    removed until none is left (the result does not depend on the order)."""
    changed = True
    while changed:
        changed = False
        for v in sorted(alive):
            if not live_arcs(arcs, alive, v, 0) or not live_arcs(arcs, alive, v, 1):
                alive.remove(v)
                changed = True


def delta(arcs: Arcs, costs: Sequence[Fraction], alive: set, v: int) -> Fraction:
    """(ii): the costs of v's out-arcs into the alive minus those of its in-arcs from them."""
    out = sum((costs[e] for e in live_arcs(arcs, alive, v, 0)), Fraction(0))
    return out - sum((costs[e] for e in live_arcs(arcs, alive, v, 1)), Fraction(0))


def eades(n: int, arcs: Arcs, costs: Sequence[Fraction]) -> list[int]:
    """F_0 of (2) by (i)-(iv): (iii) the first vertex id of the largest delta, (iv) its
    in-arcs from the alive join F_0; ascending."""
    alive, f0 = set(range(n)), []
    strip(arcs, alive)
    while alive:
        best = max(sorted(alive), key=lambda v: delta(arcs, costs, alive, v))
        f0 += live_arcs(arcs, alive, best, 1)
        alive.remove(best)
        strip(arcs, alive)
    return sorted(f0)


def igraph_eades(n: int, arcs: Arcs, costs: Sequence[Fraction]) -> list[int]:
    """F_0 of (2) as the spec defines it: igraph's Eades on float(c_e)."""
    g = igraph.Graph(n=n, edges=list(arcs), directed=True)
    return sorted(g.feedback_arc_set(weights=[float(c) for c in costs], method="eades"))


def reaches(arcs: Arcs, removed: set, start: int, goal: int) -> bool:
    """A directed path from start to goal over the arcs not in removed."""
    seen, todo = {start}, [start]
    while todo:
        x = todo.pop()
        for e, (u, v) in enumerate(arcs):
            if u == x and e not in removed and v not in seen:
                seen.add(v)
                todo.append(v)
    return goal in seen


def put_back(arcs: Arcs, costs: Sequence[Fraction], f0: list[int]) -> list[int]:
    """(3): F_0 in descending cost, ties by ascending id; each arc u -> v leaves the removed
    set iff no path v -> u avoids the arcs still removed. F ascending."""
    removed = set(f0)
    for e in sorted(f0, key=lambda e: (-costs[e], e)):
        u, v = arcs[e]
        if not reaches(arcs, removed, v, u):
            removed.remove(e)
    return sorted(removed)


def removed_oracle(
    n: int, arcs: Arcs, costs: Sequence[Fraction], f0: Callable = eades
) -> list[int]:
    """F of (2), (3); f0 = eades (dyadic costs) or igraph_eades (any costs)."""
    return put_back(arcs, costs, f0(n, arcs, costs))


def acyclic(arcs: Arcs, removed: set) -> bool:
    """G without the removed arcs has no directed cycle."""
    return not any(
        reaches(arcs, removed, v, u)
        for e, (u, v) in enumerate(arcs)
        if e not in removed
    )


def maximal(arcs: Arcs, removed: set) -> bool:
    """No removed arc u -> v can be added back without a cycle (a path v -> u)."""
    return all(reaches(arcs, removed, arcs[e][1], arcs[e][0]) for e in removed)


def random_graph(
    rng: random.Random, n: int, m: int, grid: Sequence[Fraction]
) -> tuple[list, list]:
    """m arcs over n vertices, at most one per pair of vertices, no loop (the domain of
    specs/preference-pool.md (G)), random direction; costs drawn from grid."""
    pairs = rng.sample([(u, v) for u in range(n) for v in range(u + 1, n)], m)
    arcs = [(u, v) if rng.random() < 0.5 else (v, u) for u, v in pairs]
    return arcs, [rng.choice(grid) for _ in arcs]


DYADIC = [Fraction(i, 16) for i in range(9)]  # costs 0..1/2: float64 sums exact
DECIMAL = [Fraction(i, 10) for i in range(6)]  # the costs of the starting config, s = 1
