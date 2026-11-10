"""The removed arcs F of specs/acyclic.md: Eades' feedback arc set (2), then the
put-back (3)."""

from collections.abc import Sequence
from fractions import Fraction

import igraph


def eades(
    n: int, arcs: Sequence[tuple[int, int]], costs: Sequence[Fraction]
) -> list[int]:
    """(2): F_0, the arc ids igraph's Eades heuristic removes, costs as float64."""
    graph = igraph.Graph(n=n, edges=list(arcs), directed=True)
    return graph.feedback_arc_set(weights=[float(c) for c in costs], method="eades")


def reaches(
    out: list[list[tuple[int, int]]], removed: set[int], v: int, u: int
) -> bool:
    """(3): a directed path from v to u over the arcs not removed (breadth-first)."""
    seen, queue = {v}, [v]
    for x in queue:
        for e, y in out[x]:
            if e not in removed and y not in seen:
                if y == u:
                    return True
                seen.add(y)
                queue.append(y)
    return False


def put_back(
    n: int, arcs: Sequence[tuple[int, int]], costs: Sequence[Fraction], f0: list[int]
) -> set[int]:
    """(3): F, the arcs of F_0 by descending cost, ties by ascending id, each put back
    when no path leads from its head to its tail without the arcs still removed."""
    out = [[] for _ in range(n)]
    for e, (u, v) in enumerate(arcs):
        out[u].append((e, v))
    removed = set(f0)
    for e in sorted(f0, key=lambda e: (-costs[e], e)):
        u, v = arcs[e]
        if not reaches(out, removed, v, u):
            removed.discard(e)
    return removed


def removed_arcs(
    n: int, arcs: Sequence[tuple[int, int]], costs: Sequence[Fraction]
) -> list[int]:
    """F of (2), (3) on a graph of n vertices, arc id = index in `arcs`; ascending."""
    return sorted(put_back(n, arcs, costs, eades(n, arcs, costs)))
