"""`tally_matrices` and `matrices` of src/mcpref/recognition/metrics.py against
specs/gui-masks.md (8): hand cases in exact fractions and the naive whole-set reference of
metrics_oracle.py, the labels tallied in one or several shards."""

import itertools
from fractions import Fraction as Fr

import metrics_oracle as o
import numpy as np
import pytest

from mcpref.recognition.metrics import matrices, tally_matrices

NAMES = ["p", "q", "r", "s"]
Y = ["p", "p", "q", "r", "p"]
PREDS = {  # labeller -> threshold choice -> y_hat
    "one": {
        "agreement": ["p", "q", "q", "r", "other"],
        "macro_recall": ["other"] * 5,
    },
    "two": {"agreement": ["q", "q", "q", "p", "p"], "macro_recall": list(Y)},
}
HAND = {  # choice -> class -> (|F_k|, matrix over VLM, one, two); s: assigned by no labeller
    "agreement": {
        # F_p = f0, f1, f3, f4: VLM = one on f0, f3; VLM = two on f4; one = two on f1
        "p": (
            4,
            [[1, Fr(1, 2), Fr(1, 4)], [Fr(1, 2), 1, Fr(1, 4)], [Fr(1, 4), Fr(1, 4), 1]],
        ),
        # F_q = f0, f1, f2: VLM = one on f0, f2; VLM = two on f2; one = two on f1, f2
        "q": (
            3,
            [[1, Fr(2, 3), Fr(1, 3)], [Fr(2, 3), 1, Fr(2, 3)], [Fr(1, 3), Fr(2, 3), 1]],
        ),
        "r": (1, [[1, 1, 0], [1, 1, 0], [0, 0, 1]]),  # F_r = f3
    },
    "macro_recall": {
        "p": (3, [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),  # F_p = f0, f1, f4
        "q": (1, [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),  # F_q = f2
        "r": (1, [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),  # F_r = f3
    },
}


def run(y: list, preds: dict, names: list, cuts=()) -> dict:
    """(8) of the labels tallied shard by shard, the shards cut at `cuts`."""
    acc = {}
    for a, b in itertools.pairwise((0, *cuts, len(y))):
        part = {n: {c: p[a:b] for c, p in by.items()} for n, by in preds.items()}
        tally_matrices(acc, y[a:b], part)
    return matrices(acc, names)


def shown(got: dict) -> dict:
    """class -> (n, matrix as floats) for the classes with a matrix (Interface: none for an
    empty F_k, the class absent or its matrix null)."""
    return {
        k: (m["n"], [[float(v) for v in r] for r in m["matrix"]])
        for k, m in got.items()
        if m and m.get("matrix") is not None
    }


def floats(case: dict) -> dict:
    return {k: (n, [[float(v) for v in r] for r in m]) for k, (n, m) in case.items()}


def test_matrices_hand_case():
    """(8): per choice and class, |F_k| and A_k in labeller order VLM, then `preds` in its order;
    no matrix for s."""
    got = run(Y, PREDS, NAMES)
    assert set(got) == set(HAND)
    for choice, case in HAND.items():
        assert set(got[choice]) <= set(NAMES)
        assert shown(got[choice]) == floats(case)
        for m in got[choice].values():
            if m and m.get("matrix") is not None:
                assert len(m["labellers"]) == 3 and list(m["labellers"][1:]) == [
                    "one",
                    "two",
                ]


def test_matrices_follow_preds_order():
    """(8): the labellers in `preds` order; swapping them permutes rows and columns."""
    swapped = {"two": PREDS["two"], "one": PREDS["one"]}
    perm = [0, 2, 1]
    got = shown(run(Y, swapped, NAMES, (2,))["agreement"])
    for k, (n, m) in floats(HAND["agreement"]).items():
        assert got[k] == (n, [[m[i][j] for j in perm] for i in perm])


@pytest.mark.parametrize("cuts", [(), (0, 7), (10, 10, 29), tuple(range(1, 30))])
@pytest.mark.parametrize("seed", [0, 1])
def test_matrices_over_shards_naive_reference(seed, cuts):
    """(8) on 30 random frames, three labellers besides the VLM, tallied in shards (empty ones
    included), against the naive whole-set reference."""
    rng = np.random.default_rng(seed)
    pool = NAMES + ["other"]
    y = [str(v) for v in rng.choice(pool, 30)]
    preds = {
        name: {
            c: [str(v) for v in rng.choice(pool, 30)]
            for c in ("agreement", "macro_recall")
        }
        for name in ("a", "b", "c")
    }
    got = run(y, preds, NAMES, cuts)
    for c in ("agreement", "macro_recall"):
        labellers = [y] + [preds[name][c] for name in preds]
        assert shown(got[c]) == o.matrices(NAMES, labellers)
