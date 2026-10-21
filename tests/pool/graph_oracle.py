"""(G) of specs/preference-pool.md transcribed with explicit loops: the comparison graph of
one draw and its orientation coins; its components found by a search of its own."""

import math
from fractions import Fraction

import numpy as np


def components(n: int, edges: set) -> list[list[int]]:
    """The connected components of (0..n-1, edges), ordered by their smallest vertex, each
    one's vertices ascending."""
    seen, out = set(), []
    for start in range(n):
        if start in seen:
            continue
        found, todo = {start}, [start]
        while todo:
            u = todo.pop()
            for v in [b for a, b in edges if a == u] + [a for a, b in edges if b == u]:
                if v not in found:
                    found.add(v)
                    todo.append(v)
        seen |= found
        out.append(sorted(found))
    return out


def pair(u: int, v: int) -> tuple[int, int]:
    return min(u, v), max(u, v)


def partners(n: int, r: int, g: np.random.Generator) -> set:
    """Step 1."""
    edges = set()
    for i in range(n):
        others = [x for x in range(n) if x != i]
        for q in g.permutation(n - 1)[: min(r, n - 1)]:
            edges.add(pair(i, others[q]))
    return edges


def connect(n: int, edges: set, g: np.random.Generator) -> int:
    """Step 2, in place; the number of edges added."""
    added = 0
    while len(comps := components(n, edges)) > 1:
        q0, q1 = g.permutation(len(comps))[:2]
        u = comps[q0][g.integers(len(comps[q0]))]
        v = comps[q1][g.integers(len(comps[q1]))]
        edges.add(pair(u, v))
        added += 1
    return added


def top_up(n: int, edges: set, rho: Fraction, g: np.random.Generator) -> None:
    """Step 3, in place, in exact arithmetic."""
    while len(edges) - n + 1 < math.ceil(rho * n) and len(edges) < n * (n - 1) // 2:
        free = [
            (i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges
        ]
        edges.add(free[g.integers(len(free))])


def graph(
    n: int, r: int, rho: Fraction, g: np.random.Generator
) -> list[tuple[int, int]]:
    """Steps 1-4: the comparisons (a, b) in order."""
    edges = partners(n, r, g)
    connect(n, edges, g)
    top_up(n, edges, rho, g)
    return [(i, j) if g.integers(2) == 0 else (j, i) for i, j in sorted(edges)]
