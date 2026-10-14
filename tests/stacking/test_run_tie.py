"""`main` of src/mcpref/stacking/run.py against specs/gui-stack.md ((5) ties, (6) evaluation)
on TIE_WORLD of stack_world.py, where every test frame gets p = (0, 1/2, 1/2, 0) exactly."""

import math

import numpy as np
import pytest
from progress_bars import Bar, assert_progress, states

from . import stack_oracle as so
from .stack_world import LABELS, TIE_CONFIG, TIE_WORLD


def test_tie_precondition():
    """TIE_CONFIG: under (2) with its seed, b1 < 0 and p_beta, p_gamma < 1/2 at theta_0."""
    _, b1, _, b2 = so.init(1, 1, 4, TIE_CONFIG["seed"])
    p = np.exp(so.log_softmax(b2[None]))[0]
    assert b1[0] < 0 and p[1] < 0.5 and p[2] < 0.5


def test_tie_takes_the_first_label(tie_run):
    """(5): p_f = (0, 1/2, 1/2, 0) for every test frame, z = 0 whatever x; y_hat = beta, the
    first of the tied beta, gamma, not the first label of Y."""
    assert tie_run.code == 0, tie_run.err
    records = tie_run.predictions()
    assert [r["id"] for r in records] == TIE_WORLD.test.ids
    tied = {"alpha": 0.0, "beta": 0.5, "gamma": 0.5, "other": 0.0}
    assert all(r["p"] == tied and r["predicted"] == "beta" for r in records)


def test_tie_report_train(tie_run):
    """train: |S| = 4; L(theta_0) of the reference; L(theta_1) = log 2 (lambda = 0)."""
    o = so.reference(TIE_WORLD, TIE_CONFIG)
    train = tie_run.report()["train"]
    assert train["frames"] == 4
    assert train["per_label"] == {"alpha": 0, "beta": 2, "gamma": 2, "other": 0}
    first = so.loss(o["path"][0], o["z"], o["y"], o["w"], 0.0)
    assert train["loss_initial"] == pytest.approx(first, rel=1e-5)
    assert train["loss_final"] == pytest.approx(math.log(2), rel=1e-6)


def test_tie_bar(tie_run):
    """T = 1: the bar is drawn and closed at 1 of 1."""
    assert_progress(tie_run.err)
    assert states(tie_run.err, "stack fit")[-1] == Bar("stack fit", 1, 1, "step")


def label(n_true, n_pred, precision, recall, f1, errors=()) -> dict:
    return {"n_true": n_true, "n_pred": n_pred, "precision": precision, "recall": recall,
            "f1": f1, "errors": list(errors)}  # fmt: skip


NONE = label(0, 0, None, None, None)
MISSED = {k: label(1, 0, None, 0.0, 0.0, [i]) for k, i in (("alpha", "t0"),
          ("gamma", "t2"), ("other", "t3"))}  # fmt: skip
TEST = {  # gui-masks (7) of y = alpha, beta, gamma, other, beta; y_hat = beta
    "agreement": 2 / 5,
    "macro_recall": 1 / 4,  # recalls 0, 1, 0, 0
    "per_label": {
        **MISSED,
        "beta": label(2, 5, 2 / 5, 1.0, 4 / 7),
    },  # f1 = 2 TP / (n + n)
    "confusion": {k: {"beta": 1 + (k == "beta")} for k in LABELS},
    "by_flag": {
        "true": {  # t0, t1, t4: alpha, beta, beta
            "agreement": 2 / 3,
            "macro_recall": 1 / 2,
            "per_label": {
                "alpha": MISSED["alpha"],
                "beta": label(2, 3, 2 / 3, 1.0, 4 / 5),
                "gamma": NONE,
                "other": NONE,
            },
            "confusion": {"alpha": {"beta": 1}, "beta": {"beta": 2}},
        },
        "false": {  # t2, t3: gamma, other
            "agreement": 0.0,
            "macro_recall": 0.0,
            "per_label": {
                "alpha": NONE,
                "beta": label(0, 2, 0.0, None, 0.0),
                "gamma": MISSED["gamma"],
                "other": MISSED["other"],
            },
            "confusion": {"gamma": {"beta": 1}, "other": {"beta": 1}},
        },
    },
}


def normal(part: dict) -> dict:
    """One set of gui-masks (7): errors sorted, confusion without zero cells."""
    per_label = {
        k: {**v, "errors": sorted(v["errors"])} for k, v in part["per_label"].items()
    }
    rows = {y: {p: n for p, n in r.items() if n} for y, r in part["confusion"].items()}
    out = {k: part[k] for k in ("agreement", "macro_recall")}
    out |= {"per_label": per_label, "confusion": {y: r for y, r in rows.items() if r}}
    if "by_flag" in part:
        out["by_flag"] = {g: normal(v) for g, v in part["by_flag"].items()}
    return out


def test_tie_report_test(tie_run):
    """(6): test = gui-masks (7) of y_hat on the test frames, with by_flag by g_f."""
    test = tie_run.report()["test"]
    assert set(test) == {
        "agreement",
        "macro_recall",
        "per_label",
        "confusion",
        "by_flag",
    }
    assert set(test["by_flag"]) == {"true", "false"}
    assert normal(test) == TEST
