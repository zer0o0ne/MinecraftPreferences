"""(G) of specs/preference-pool.md: the comparison graph of a draw's n clips, with its
graph generator g_k."""

import math
from fractions import Fraction

import numpy as np

from mcpref.clips.sample import uniform
from mcpref.graphs.components import components

Edges = set[tuple[int, int]]
"""E: unordered pairs {i, j} as (i, j), i < j."""


def edge(i: int, j: int) -> tuple[int, int]:
    """The unordered pair {i, j}."""
    return min(i, j), max(i, j)


def partners(n: int, r: int, g: np.random.Generator) -> Edges:
    """(G) 1: per i = 0..n-1, the partners O_i[q] for q in g.permutation(n - 1)[:min(r,
    n - 1)], O_i = 0..n-1 without i, ascending."""
    E = set()
    for i in range(n):
        for q in g.permutation(n - 1)[: min(r, n - 1)].tolist():
            E.add(edge(i, q if q < i else q + 1))
    return E


def components_of(n: int, E: Edges) -> list[list[int]]:
    """(G) 2: the connected components of (0..n-1, E) ordered by their smallest vertex,
    each one's vertices ascending."""
    labels = components(list(range(n)), list(E))
    found = [[] for _ in range(max(labels) + 1)]
    for v, j in enumerate(labels):
        found[j].append(v)
    return found


def connect(n: int, E: Edges, g: np.random.Generator) -> None:
    """(G) 2: while several components, {u, v} added: q_0, q_1 = g.permutation(<count>)
    [:2], u drawn in component q_0, then v in component q_1."""
    while len(found := components_of(n, E)) > 1:
        q0, q1 = g.permutation(len(found))[:2]
        u = found[q0][uniform(g, len(found[q0]))]
        v = found[q1][uniform(g, len(found[q1]))]
        E.add(edge(u, v))


def top_up(n: int, rho: Fraction, E: Edges, g: np.random.Generator) -> None:
    """(G) 3: while |E| - n + 1 < ceil(rho n) (exact) and E is not complete, the missing
    pair at g.integers(<their count>) in lexicographic order added."""
    cycles = math.ceil(rho * n)
    while len(E) - n + 1 < cycles and len(E) < n * (n - 1) // 2:
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in E]
        E.add(missing[uniform(g, len(missing))])


def comparisons(
    n: int, r: int, rho: Fraction, g: np.random.Generator
) -> list[tuple[int, int]]:
    """(G) 1-4, n >= 2: the pairs of E in lexicographic order, each oriented by
    o = g.integers(2): (i, j) if o = 0, else (j, i)."""
    E = partners(n, r, g)
    connect(n, E, g)
    top_up(n, rho, E, g)
    return [(i, j) if uniform(g, 2) == 0 else (j, i) for i, j in sorted(E)]
