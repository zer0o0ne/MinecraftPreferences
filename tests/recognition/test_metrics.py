"""`tally` and `evaluation` of src/mcpref/recognition/metrics.py against specs/gui-masks.md
(7): hand cases in exact fractions and the naive whole-set reference of metrics_oracle.py,
the frames tallied in one or several shards; (8) in test_metrics_matrices.py."""

import itertools
from collections.abc import Mapping
from fractions import Fraction as Fr

import metrics_oracle as o
import numpy as np
import pytest
from metrics_oracle import normal, num

from mcpref.recognition.metrics import CRITERIA, evaluation, tally

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


def run(frames: list, cuts=()) -> dict:
    """(7) of `frames` tallied shard by shard, the shards cut at `cuts`."""
    acc = {}
    for a, b in itertools.pairwise((0, *cuts, len(frames))):
        ids, y, y_hat, g = ([f[i] for f in frames[a:b]] for i in range(4))
        tally(acc, y, y_hat, ids, g)
    return evaluation(acc, LABELS)


def row(n_true, n_pred, precision, recall, f1, errors) -> dict:
    ratios = {"precision": num(precision), "recall": num(recall), "f1": num(f1)}
    return {"n_true": n_true, "n_pred": n_pred, **ratios, "errors": errors}


def test_evaluation_hand_case():
    """(7): a: TP 1 of 3 true, 3 predicted, F1 = 2/6; c never true (recall null, F1 0/1);
    d never predicted (precision null); x absent (all null); agreement 3/8; macro recall over
    the labels present in y (a, b, d, other) = (1/3 + 1/2 + 0 + 1/2) / 4 = 1/3."""
    got = run(FRAMES)
    assert set(got) == SET_KEYS | {"by_flag"}
    assert set(got["per_label"]) == set(LABELS)
    assert all(set(v) == LABEL_KEYS for v in got["per_label"].values())
    per_label = {
        "a": row(3, 3, Fr(1, 3), Fr(1, 3), Fr(1, 3), ["f2", "f4"]),
        "b": row(2, 2, Fr(1, 2), Fr(1, 2), Fr(1, 2), ["f6"]),
        "c": row(0, 1, Fr(0), None, Fr(0), []),
        "d": row(1, 0, None, Fr(0), Fr(0), ["f8"]),
        "x": row(0, 0, None, None, None, []),
        "other": row(2, 2, Fr(1, 2), Fr(1, 2), Fr(1, 2), ["f5"]),
    }
    confusion = {
        "a": {"a": 1, "b": 1, "other": 1},
        "b": {"b": 1, "c": 1},
        "other": {"a": 1, "other": 1},
        "d": {"a": 1},
    }
    expected = {"agreement": num(Fr(3, 8)), "macro_recall": num(Fr(1, 3))}
    assert normal(got) == {**expected, "per_label": per_label, "confusion": confusion}


def test_evaluation_by_flag_hand_case():
    """(7) split by g: true = f1, f3, f4, f7 (agreement 3/4, macro recall (1/2 + 1 + 1) / 3 =
    5/6; other: precision 1/2, recall 1, F1 2/3); false = f2, f5, f6, f8 (agreement 0)."""
    by_flag = run(FRAMES)["by_flag"]
    assert set(by_flag) == {"true", "false"}
    assert all(set(part) == SET_KEYS for part in by_flag.values())
    true, false = normal(by_flag["true"]), normal(by_flag["false"])
    assert (true["agreement"], true["macro_recall"]) == (num(Fr(3, 4)), num(Fr(5, 6)))
    assert true["per_label"]["other"] == row(1, 2, Fr(1, 2), Fr(1), Fr(2, 3), [])
    assert true["per_label"]["d"] == row(0, 0, None, None, None, [])
    assert true["confusion"] == {
        "a": {"a": 1, "other": 1},
        "b": {"b": 1},
        "other": {"other": 1},
    }
    assert (false["agreement"], false["macro_recall"]) == (0.0, 0.0)
    assert false["per_label"]["c"] == row(0, 1, Fr(0), None, Fr(0), [])
    for flag, part in (("true", true), ("false", false)):
        rows = [f for f in FRAMES if f[3] == (flag == "true")]
        ids, y, y_hat = ([f[i] for f in rows] for i in range(3))
        assert part == o.metrics(ids, y, y_hat, LABELS)


def test_evaluation_empty_sets():
    """(7): agreement and macro recall null on an empty set, here the part with g false and the
    whole of an empty input; counts 0, every ratio null."""
    got = run([f for f in FRAMES if f[3]])
    assert normal(got["by_flag"]["false"]) == {
        "agreement": None,
        "macro_recall": None,
        "per_label": {k: row(0, 0, None, None, None, []) for k in LABELS},
        "confusion": {},
    }
    for empty in (run([]), evaluation({}, LABELS)):
        assert (empty["agreement"], empty["macro_recall"]) == (None, None)
        assert normal(empty)["per_label"]["a"] == row(0, 0, None, None, None, [])


def random_frames(seed: int, n: int = 40) -> list:
    rng = np.random.default_rng(seed)
    return [
        (
            f"id{i}",
            str(rng.choice(LABELS[:4] + ["other"])),
            str(rng.choice(LABELS)),
            bool(g),
        )
        for i, g in enumerate(rng.integers(0, 2, n))
    ]


SHARDS = [(), (3,), (0, 0, 17), (5, 5, 13, 39), tuple(range(1, 40))]


@pytest.mark.parametrize("cuts", SHARDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluation_over_shards_naive_reference(seed, cuts):
    """(7) of 40 random frames tallied in shards (empty ones included) against the naive
    whole-set reference, as a whole and by flag."""
    frames = random_frames(seed)
    got = run(frames, cuts)
    ids, y, y_hat = ([f[i] for f in frames] for i in range(3))
    assert normal(got) == o.metrics(ids, y, y_hat, LABELS)
    for flag in (True, False):
        rows = [f for f in frames if f[3] == flag]
        sub = ([f[i] for f in rows] for i in range(3))
        assert normal(got["by_flag"][str(flag).lower()]) == o.metrics(*sub, LABELS)


@pytest.mark.parametrize("cuts", SHARDS)
def test_errors_in_frame_order_over_shards(cuts):
    """(7): the ids of the frames with y = k and y_hat != k, in the order tallied."""
    frames = random_frames(5)[::-1]  # ids not sorted
    got = run(frames, cuts)["per_label"]
    for k in LABELS:
        assert list(got[k]["errors"]) == [i for i, a, b, _ in frames if a == k != b]


def test_criteria_threshold_choices_in_order():
    """Interface: CRITERIA maps the two threshold choices of (6), in the order agreement,
    macro_recall (the criteria's own signature is not part of the Interface)."""
    assert isinstance(CRITERIA, Mapping)
    assert list(CRITERIA) == ["agreement", "macro_recall"]
