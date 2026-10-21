"""`components` of src/mcpref/graphs/components.py (moved from pretrain/items.py by
specs/preference-pool.md, Revisions; Interface): for each vertex, the index of its connected
component, the components ordered by their first vertex in the order given — the groups of
specs/train-index.md (index_oracle.py) and the components of (G) 2 (graph_oracle.py) on
list(range(n))."""

import numpy as np
import pytest
from pool.graph_oracle import components as by_smallest
from pretrain.index_oracle import groups

from mcpref.graphs.components import components


@pytest.mark.parametrize(
    "eps, edges",
    [
        (["a", "b", "c"], []),
        (["e2", "e0", "e1", "e3"], [("e1", "e3"), ("e0", "e1")]),
        (["x", "y", "z", "w", "v"], [("v", "x"), ("z", "w"), ("w", "z"), ("y", "y")]),
        ([("t", 1), ("t", 2), 7], [(7, ("t", 1))]),
    ],
)  # fmt: skip
def test_hashable_vertices_numbered_by_first_appearance(eps, edges):
    want = groups(eps, [(None, a, b) for a, b in edges])
    assert components(eps, edges) == [want[e] for e in eps]


@pytest.mark.parametrize("seed", range(5))
def test_integer_vertices_ordered_by_smallest_vertex(seed):
    """(G) 2: vertices 0..n-1, random integer pairs; component q of graph_oracle.py is
    the one holding the vertices numbered q."""
    rng = np.random.default_rng(seed)
    n = 15
    edges = sorted({tuple(sorted(map(int, rng.choice(n, 2, replace=False)))) for _ in range(9)})  # fmt: skip
    got = components(list(range(n)), edges)
    for q, comp in enumerate(by_smallest(n, set(edges))):
        assert {got[v] for v in comp} == {q}
