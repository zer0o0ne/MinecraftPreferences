"""(G) of specs/preference-pool.md: `comparisons` against the transcription of
graph_oracle.py with Generator(PCG64([seed, 5, k])) — random partners, components joined
until connected, the cycle top-up to ceil(rho n) in exact arithmetic or the complete
graph, the orientation coins — and the generator left where the transcription leaves it."""

import math
from fractions import Fraction

import numpy as np
import pytest

from mcpref.pool.graph import comparisons

from . import graph_oracle as oracle


def gen(seed: int, k: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([seed, 5, k]))


def both(n: int, r: int, rho: str, seed: int, k: int = 0) -> list:
    """comparisons and the transcription on equal generators; the generators end equal."""
    g, h = gen(seed, k), gen(seed, k)
    got = comparisons(n, r, Fraction(rho), g)
    want = oracle.graph(n, r, Fraction(rho), h)
    assert [tuple(x) for x in got] == want
    assert g.bit_generator.state == h.bit_generator.state
    return got


@pytest.mark.parametrize("n, r, rho", [(3, 1, "0"), (4, 2, "0.5"), (7, 1, "0.5"), (9, 2, "1.25"), (12, 3, "0.3"), (20, 1, "0.07")])  # fmt: skip
@pytest.mark.parametrize("seed, k", [(0, 0), (0, 1), (7, 3), (2**32 - 1, 12)])
def test_comparisons_follow_the_transcription(n, r, rho, seed, k):
    got = both(n, r, rho, seed, k)
    assert all(isinstance(x, int) for p in got for x in p)
    pairs = [(min(p), max(p)) for p in got]
    assert pairs == sorted(set(pairs)) and all(a != b for a, b in got)
    assert len(oracle.components(n, set(pairs))) == 1


def test_two_clips_one_comparison_either_way():
    """n = 2: the edge {0, 1}; its coin orients it."""
    got = {
        tuple(map(tuple, both(2, r, "0.5", seed))) for seed in range(12) for r in (1, 3)
    }
    assert got == {((0, 1),), ((1, 0),)}


@pytest.mark.parametrize("r", [4, 9])
def test_r_at_least_n_minus_1_gives_the_complete_graph(r):
    """Step 1 links every pair; no draw for steps 2 and 3; ten coins."""
    for seed in range(3):
        got = both(5, r, "3", seed)
        assert sorted((min(p), max(p)) for p in got) == [(i, j) for i in range(5) for j in range(i + 1, 5)]  # fmt: skip


def test_several_components_are_joined():
    """n = 30, r = 1: seeds whose partners leave at least three components."""
    seeds = []
    for seed in range(200):
        g = gen(seed)
        edges = oracle.partners(30, 1, g)
        if len(oracle.components(30, edges)) >= 3:
            seeds.append(seed)
    assert len(seeds) >= 3
    for seed in seeds[:3]:
        both(30, 1, "0", seed)


def test_top_up_stops_at_ceil_rho_n_in_exact_arithmetic():
    """rho 0.07, n 100: ceil(7) = 7 independent cycles, 106 edges (in floats ceil gives 8);
    steps 1-2 leave fewer."""
    assert math.ceil(0.07 * 100) == 8
    for seed in range(3):
        g = gen(seed)
        edges = oracle.partners(100, 1, g)
        oracle.connect(100, edges, g)
        assert len(edges) - 99 < 7
        assert len(both(100, 1, "0.07", seed)) == 106


def test_top_up_stops_at_the_complete_graph():
    """rho 10 asks for 50 cycles; n = 5 has 10 pairs."""
    for seed in range(3):
        assert len(both(5, 1, "10", seed)) == 10


def test_no_top_up_for_rho_0():
    for seed in range(3):
        g = gen(seed)
        edges = oracle.partners(10, 1, g)
        oracle.connect(10, edges, g)
        assert len(both(10, 1, "0", seed)) == len(edges)
