"""`removed_arcs` of specs/acyclic.md (Interface, `graphs/feedback.py`): F of (2), (3) on hand
graphs (hand-derived), against the in-test transcription of feedback_oracle.py on random
graphs (dyadic costs: exact; decimal costs: igraph's float64 semantics, the definition of
(2)), the properties (3) states and the disjoint union of (3')."""

import random
from fractions import Fraction

import pytest

from mcpref.graphs.feedback import removed_arcs

from .feedback_oracle import (
    DECIMAL,
    DYADIC,
    acyclic,
    igraph_eades,
    maximal,
    random_graph,
    removed_oracle,
)


def fr(*xs) -> list[Fraction]:
    return [Fraction(x) for x in xs]


THREE_CYCLE = (3, [(0, 1), (1, 2), (2, 0)], fr("1/2", "1/4", "1/8"))
ZERO_COST = (3, [(0, 1), (1, 2), (2, 0)], fr(0, "1/2", "1/2"))

# (name, n, arcs, costs, F); delta(v) = out-costs minus in-costs among the alive vertices.
HAND = [
    # a DAG with a diamond, zero costs and an isolated vertex 5: no cycle, F_0 empty
    ("dag", 6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], fr(0, "1/2", "1/8", 0, "1/4"), []),
    ("empty", 0, [], [], []),
    # delta = 3/8, -1/4, -1/8: v* = 0, its in-arc 2 removed; 0 -> 1 -> 2 keeps it out
    ("three-cycle", *THREE_CYCLE, [2]),
    # delta all 0: v* = vertex 0 (lowest id), its in-arc 1 (2 -> 0)
    ("exact tie", 3, [(1, 2), (2, 0), (0, 1)], fr("1/4", "1/4", "1/4"), [1]),
    # the zero-cost arc 0 still counts: 1 is no source; delta = -1/2, 1/2, 0: v* = 1, arc 0
    ("zero-cost arc", *ZERO_COST, [0]),
    # cycles 0-1-2, 0-1-3 share arc 0; delta = -7/8, 7/8, 0, 0: v* = 1, arc 0 breaks both
    ("shared arc removed", 4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)],
     fr("1/8", "1/2", "1/2", "1/2", "1/2"), [0]),
    # delta = -1/2, -1/4, 3/8, 3/8: v* = 2 (tie), arc 1; then 0-1-3: delta 0, -3/8, 3/8:
    # v* = 3, arc 3; both kept out (2 -> 0 -> 1, 3 -> 0 -> 1)
    ("shared arc kept", 4, [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)],
     fr("1/2", "1/8", "1/2", "1/8", "1/2"), [1, 3]),
    # delta = 1/4, -3/8, -1/4, 3/8: v* = 3, arc 0; then 0-2-1: v* = 0, arc 1; F_0 = {0, 1};
    # put back by cost: arc 1 stays (0 -> 2 -> 1), then arc 0 back (3 -> 1 only, 1 -> 0 out)
    ("put back", 4, [(0, 3), (1, 0), (3, 1), (2, 1), (0, 2)],
     fr(0, "1/8", "3/8", "1/8", "3/8"), [1]),
    # F_0 = {2, 4, 7} (v* = 3, 0, 1); order 2, 7 (1/8, by id), 4 (0): 2 stays (1 -> 4 -> 2),
    # 7 back (3 -> 1 -> 4 -> 2 ends), 4 stays (0 -> 3 -> 1); by id alone 4 would come back
    ("put back by cost", 5, [(3, 1), (1, 4), (2, 1), (0, 2), (1, 0), (4, 2), (0, 4), (0, 3)],
     fr("1/2", "1/4", "1/8", 0, 0, "1/4", "1/8", "1/8"), [2, 4]),
    # F_0 = {0, 1, 6} (v* = 1 (tie with 4), 4, 2); order 1 (1/4), 0, 6 (0, by id): 1 stays
    # (2 -> 3 -> 0), 0 back (4 -> 3 -> 0 ends), 6 stays (1 -> 2 -> 4); 6 first: 6 back
    ("put back by id", 5, [(2, 4), (0, 2), (2, 3), (1, 2), (3, 0), (4, 3), (4, 1)],
     fr(0, "1/4", "1/2", "3/8", "3/8", "3/8", 0), [1, 6]),
    # float64: delta(1) = 0.3 - 0.2 = 0.09999999999999998 < delta(2) = 0.2 - 0.1 = 0.1,
    # so v* = 2 and arc 2 (the exact tie would pick vertex 1 and arc 1)
    ("float64 tie", 3, [(1, 0), (2, 1), (0, 2)], fr("3/10", "1/5", "1/10"), [2]),
    # the three-cycle on 0, 2, 4 and the zero-cost one on 1, 3, 5, arcs interleaved: global
    # v* = 3 (delta 1/2), arc 1; then v* = 0 (3/8), arc 4
    ("components", 6, [(0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)],
     fr("1/2", 0, "1/4", "1/2", "1/8", "1/2"), [1, 4]),
]  # fmt: skip


@pytest.mark.parametrize(
    "name, n, arcs, costs, expected", HAND, ids=[h[0] for h in HAND]
)
def test_hand_graphs(name, n, arcs, costs, expected):
    """(2), (3) on hand graphs: F, ascending."""
    assert removed_arcs(n, arcs, costs) == expected


def test_hand_graphs_agree_with_the_transcription():
    """The hand results are those of the transcription (the float64 tie: of igraph's
    semantics; the others' F_0 also of the exact one)."""
    for name, n, arcs, costs, expected in HAND:
        assert removed_oracle(n, arcs, costs, igraph_eades) == expected, name
        exact = [1] if name == "float64 tie" else expected
        assert removed_oracle(n, arcs, costs) == exact, name


def random_cases(seed: int, count: int, grid: list) -> list:
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        n = rng.randrange(2, 11)
        cases.append(
            (n, *random_graph(rng, n, rng.randrange(n * (n - 1) // 2 + 1), grid))
        )
    return cases


def test_dyadic_costs_match_the_exact_transcription():
    """(2), (3) on random graphs with dyadic costs (float64 exact): F of (i)-(iv) in exact
    arithmetic with the lowest-id tie-break, then the put-back."""
    for n, arcs, costs in random_cases(0, 400, DYADIC):
        assert removed_arcs(n, arcs, costs) == removed_oracle(n, arcs, costs)


def test_decimal_costs_follow_float64_semantics():
    """(2) with igraph's float64 semantics as the definition, on the costs of the starting
    config (multiples of 0.1): F of igraph's F_0 put back; the sample holds graphs whose
    exact lowest-id F differs."""
    cases = random_cases(1, 400, DECIMAL)
    differ = 0
    for n, arcs, costs in cases:
        expected = removed_oracle(n, arcs, costs, igraph_eades)
        assert removed_arcs(n, arcs, costs) == expected
        differ += expected != removed_oracle(n, arcs, costs)
    assert differ >= 5


@pytest.mark.parametrize("grid", [DYADIC, DECIMAL], ids=["dyadic", "decimal"])
def test_properties(grid):
    """(3): F ascending, distinct arc ids; G without F acyclic; no arc of F can be added
    back without a cycle."""
    for n, arcs, costs in random_cases(2, 300, grid):
        f = removed_arcs(n, arcs, costs)
        assert f == sorted(set(f)) and all(0 <= e < len(arcs) for e in f)
        assert acyclic(arcs, set(f)) and maximal(arcs, set(f))


def interleave(rng: random.Random, sizes: list[int]) -> list[list[int]]:
    """For each part, the positions of its items in a random merge keeping their order."""
    labels = [i for i, size in enumerate(sizes) for _ in range(size)]
    rng.shuffle(labels)
    places = [[] for _ in sizes]
    for p, i in enumerate(labels):
        places[i].append(p)
    return places


@pytest.mark.parametrize("grid", [DYADIC, DECIMAL], ids=["dyadic", "decimal"])
def test_disjoint_union(grid):
    """(3'): F of a disjoint union, vertices and arcs of each component merged in their
    relative order, is the union of the components' F."""
    rng = random.Random(3)
    for _ in range(150):
        parts = random_cases(rng.randrange(10**6), rng.randrange(2, 5), grid)
        vertex = interleave(rng, [n for n, _, _ in parts])
        arc = interleave(rng, [len(arcs) for _, arcs, _ in parts])
        m = sum(len(arcs) for _, arcs, _ in parts)
        union, costs, expected = [None] * m, [None] * m, []
        for i, (n, arcs, cs) in enumerate(parts):
            for e, ((u, v), c) in enumerate(zip(arcs, cs)):
                union[arc[i][e]] = (vertex[i][u], vertex[i][v])
                costs[arc[i][e]] = c
            expected += [arc[i][e] for e in removed_arcs(n, arcs, cs)]
        n = sum(n for n, _, _ in parts)
        assert removed_arcs(n, union, costs) == sorted(expected)
