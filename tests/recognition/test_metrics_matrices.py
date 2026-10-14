"""`matrices` of src/mcpref/recognition/metrics.py against specs/gui-masks.md (8): hand cases
in exact fractions and the naive reference of metrics_oracle.py."""

from fractions import Fraction as Fr

import metrics_oracle as o
import numpy as np
import pytest

from mcpref.recognition.metrics import matrices

LABELS = ["a", "b", "c", "d", "x", "other"]
FRAMES = [  # id, y, y_hat, g
    ("f1", "a", "a", True),
    ("f2", "a", "b", False),
    ("f3", "b", "b", True),
    ("f4", "a", "other", True),
    ("f5", "other", "a", False),
    ("f6", "b", "c", False),
    ("f7", "other", "other", True),
    ("f8", "d", "a", False),
]
SET_KEYS = {"agreement", "macro_recall", "per_label", "confusion"}
LABEL_KEYS = {"n_true", "n_pred", "precision", "recall", "f1", "errors"}


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
    got = matrices(Y, PREDS, NAMES)
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
    got = shown(matrices(Y, swapped, NAMES)["agreement"])
    for k, (n, m) in floats(HAND["agreement"]).items():
        assert got[k] == (n, [[m[i][j] for j in perm] for i in perm])


@pytest.mark.parametrize("seed", [0, 1])
def test_matrices_naive_reference(seed):
    """(8) on 30 random frames, three labellers besides the VLM, against masks_oracle.matrices."""
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
    got = matrices(y, preds, NAMES)
    for c in ("agreement", "macro_recall"):
        labellers = [y] + [preds[name][c] for name in preds]
        assert shown(got[c]) == o.matrices(NAMES, labellers)
