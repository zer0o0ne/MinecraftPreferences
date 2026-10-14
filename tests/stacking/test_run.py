"""`main` of src/mcpref/stacking/run.py against specs/gui-stack.md ((5), (6), Interface:
Produces, progress bar) on NAN_WORLD of stack_world.py, against the float64 reference."""

import numpy as np
import pytest
import torch
from progress_bars import Bar, assert_progress, screen, states
from torch.testing import assert_close

from . import stack_oracle as so
from .stack_world import CHOICES, CONFIG, LABELS, NAN_WORLD, TIE_WORLD, VARIANTS, K


def test_report_train(nan_run):
    """train: |S|, n_y over Y (gamma, absent from S, as 0), L(theta_0), L(theta_T)."""
    assert nan_run.code == 0, nan_run.err
    o = so.reference(NAN_WORLD, CONFIG)
    report = nan_run.report()
    assert set(report) == {"train", "test", "agreement_matrices"}
    train = report["train"]
    assert set(train) == {"frames", "per_label", "loss_initial", "loss_final"}
    assert train["frames"] == 7
    assert train["per_label"] == {"alpha": 4, "beta": 2, "gamma": 0, "other": 1}
    losses = [
        so.loss(o["path"][i], o["z"], o["y"], o["w"], CONFIG["l2"]) for i in (0, -1)
    ]
    got = [train["loss_initial"], train["loss_final"]]
    assert all(type(v) is float for v in got) and got == pytest.approx(losses, rel=1e-5)


def test_model_file(nan_run):
    """model.pt: theta_T (float32, detached, CPU) of the float64 reference of (2), (4);
    mu, s of S (float64; NaN ignored; the constant column s = 0); columns, labels."""
    o = so.reference(NAN_WORLD, CONFIG)
    model = nan_run.model()
    assert set(model) == {"params", "mu", "s", "columns", "labels"}
    assert model["columns"] == NAN_WORLD.columns and model["labels"] == LABELS
    for p, want in zip(model["params"], o["path"][-1], strict=True):
        assert p.dtype == torch.float32 and p.device.type == "cpu"
        assert not p.requires_grad
        assert_close(p, torch.tensor(want, dtype=torch.float32))
    assert model["mu"].dtype == model["s"].dtype == torch.float64
    assert_close(model["mu"], torch.tensor(o["mu"]))
    assert_close(model["s"], torch.tensor(o["s"]))
    assert model["s"][3].item() == 0.0


def first_argmax(p: dict) -> str:
    """(5): the first label of Y with the largest probability."""
    return max(LABELS, key=lambda k: (p[k], -LABELS.index(k)))


def test_predictions(nan_run):
    """predictions.jsonl: the test frames in file order with y_f, p_f = softmax(l) at the saved
    theta_T (z of the test frames from mu, s of S), y_hat_f the first label with the largest
    p_f."""
    o = so.reference(NAN_WORLD, CONFIG)
    theta = [p.double().numpy() for p in nan_run.model()["params"]]
    want = np.exp(so.log_softmax(so.logits(theta, o["z_te"])))
    records = nan_run.predictions()
    assert [(r["id"], r["label"]) for r in records] == list(
        zip(NAN_WORLD.test.ids, NAN_WORLD.test.y)
    )
    for r, row in zip(records, want):
        assert set(r) == {"id", "label", "predicted", "p"} and set(r["p"]) == set(
            LABELS
        )
        got = [r["p"][k] for k in LABELS]
        assert got == pytest.approx(row.tolist(), rel=1e-5, abs=1e-8)
        assert r["predicted"] == first_argmax(r["p"])


def test_progress_bar(nan_run):
    """Bar `stack fit` (unit step): drawn from 0, total T, closed at T, no postfix."""
    t = CONFIG["steps"]
    assert_progress(nan_run.err)
    assert states(nan_run.err, "stack fit")[0] == Bar("stack fit", 0, t, "step")
    bars = [b for b in screen(nan_run.err) if isinstance(b, Bar)]
    assert bars == [Bar("stack fit", t, t, "step")]


def test_report_md_title(nan_run):
    lines = (nan_run.out / "report.md").read_text().splitlines()
    assert lines[0] == "# GUI stack report"


def shown(got: dict) -> dict:
    """Classes with a matrix: (|F_k|, matrix); no matrix for an empty F_k (the class absent
    or its matrix null)."""
    return {
        k: (m["n"], [[float(v) for v in r] for r in m["matrix"]])
        for k, m in got.items()
        if m and m.get("matrix") is not None
    }


@pytest.mark.parametrize("which", ["nan", "tie"])
def test_agreement_matrices(which, nan_run, tie_run):
    """(6): gui-masks (8) per threshold choice; labellers VLM, (a)-(f) with their
    test_pred_<variant>_<choice>, then stack with y_hat under both choices."""
    world, run = {"nan": (NAN_WORLD, nan_run), "tie": (TIE_WORLD, tie_run)}[which]
    y_hat = [r["predicted"] for r in run.predictions()]
    got = run.report()["agreement_matrices"]
    assert set(got) == set(CHOICES)
    for c in CHOICES:
        labellers = [world.test.y]
        labellers += [world.preds[f"test_pred_{v}_{c}"] for v in VARIANTS] + [y_hat]
        assert shown(got[c]) == so.matrices(K, labellers)
        for m in got[c].values():
            if m and m.get("matrix") is not None:
                assert len(m["labellers"]) == 8
                assert list(m["labellers"][1:]) == [*VARIANTS, "stack"]
