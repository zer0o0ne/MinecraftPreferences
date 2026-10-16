"""`main` of src/mcpref/stacking/run.py against specs/gui-stack.md ((5), (6), Behaviour (R),
(M), (P), Produces, progress bars) on NAN_WORLD of stack_world.py, against the float64
reference."""

import hashlib
import json

import numpy as np
import pytest
import torch
import yaml
from progress_bars import Bar, assert_progress, screen
from torch.testing import assert_close

from . import stack_oracle as so
from .stack_files import EMPTY, World, run_main
from .stack_world import (
    CHOICES,
    CONFIG,
    LABELS,
    NAN_WORLD,
    TIE_WORLD,
    VARIANTS,
    K,
    reference,
)

S_SHARDS = [("train", 0), ("train", 2), ("validation", 0), ("validation", 1)]


def test_report_train(nan_run):
    """train: |S|, n_y over Y (gamma, absent from S, as 0), L(theta_0), L(theta_T) of (3)."""
    assert nan_run.code == 0, nan_run.err
    o = reference(NAN_WORLD, CONFIG)
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
    """model.pt: theta_T (float32, detached, CPU) of the float64 reference of (2), (4); mu, s
    of S (float64; NaN ignored; the constant column s = 0); columns, labels; the losses."""
    o = reference(NAN_WORLD, CONFIG)
    model = nan_run.model()
    assert set(model) == {"params", "mu", "s", "columns", "labels", "loss_initial",
                          "loss_final"}  # fmt: skip
    assert model["columns"] == NAN_WORLD.columns and model["labels"] == LABELS
    for p, want in zip(model["params"], o["path"][-1], strict=True):
        assert p.dtype == torch.float32 and p.device.type == "cpu"
        assert not p.requires_grad
        assert_close(p, torch.tensor(want, dtype=torch.float32))
    assert model["mu"].dtype == model["s"].dtype == torch.float64
    assert_close(model["mu"], torch.tensor(o["mu"]))
    assert_close(model["s"], torch.tensor(o["s"]))
    assert model["s"][3].item() == 0.0
    train = nan_run.report()["train"]
    assert (model["loss_initial"], model["loss_final"]) == (
        train["loss_initial"],
        train["loss_final"],
    )


def first_argmax(p: dict) -> str:
    """(5): the first label of Y with the largest probability."""
    return max(LABELS, key=lambda k: (p[k], -LABELS.index(k)))


def test_prediction_shards(nan_run):
    """predictions/<name(r)>/predictions.jsonl per test shard, the empty one included: its
    frames in order with y_f, p_f = softmax(l) at the saved theta_T (z from mu, s of S), and
    y_hat_f the first label with the largest p_f."""
    o = reference(NAN_WORLD, CONFIG)
    theta = [p.double().numpy() for p in nan_run.model()["params"]]
    assert [f.parent.name for f in nan_run.prediction_files()] == ["000000", "000001",
                                                                    "000002"]  # fmt: skip
    for records, shard, z in zip(
        nan_run.predictions(), NAN_WORLD.parts["test"], o["z_te"]
    ):
        assert [(r["id"], r["label"]) for r in records] == list(zip(shard.ids, shard.y))
        want = np.exp(so.log_softmax(so.logits(theta, z)))
        for r, row in zip(records, want):
            assert set(r) == {"id", "label", "predicted", "p"} and set(r["p"]) == set(
                LABELS
            )
            got = [r["p"][k] for k in LABELS]
            assert got == pytest.approx(row.tolist(), rel=1e-5, abs=1e-8)
            assert r["predicted"] == first_argmax(r["p"])
    assert nan_run.prediction_files()[1].read_text() == ""


def test_report_md_title(nan_run):
    lines = (nan_run.out / "report.md").read_text().splitlines()
    assert lines[0] == "# GUI stack report"


def shown(got: dict) -> dict:
    """Classes with a matrix: (|F_k|, matrix); none for an empty F_k."""
    return {
        k: (m["n"], [[float(v) for v in r] for r in m["matrix"]])
        for k, m in got.items()
        if m and m.get("matrix") is not None
    }


@pytest.mark.parametrize("which", ["nan", "tie"])
def test_agreement_matrices(which, nan_run, tie_run):
    """(6): gui-masks (8) per threshold choice over all test shards; labellers VLM, (a)-(f)
    with `pred` of the test shards, then stack with y_hat under both choices."""
    world, run = {"nan": (NAN_WORLD, nan_run), "tie": (TIE_WORLD, tie_run)}[which]
    frames = [(s, f) for s in world.parts["test"] for f in range(len(s.ids))]
    y_hat = [r["predicted"] for records in run.predictions() for r in records]
    got = run.report()["agreement_matrices"]
    assert set(got) == set(CHOICES)
    for c, choice in enumerate(CHOICES):
        labellers = [[s.y[f] for s, f in frames]]
        labellers += [[s.pred[f][v][c] for s, f in frames] for v in range(6)] + [y_hat]
        assert shown(got[choice]) == so.matrices(K, labellers)
        for m in got[choice].values():
            if m and m.get("matrix") is not None:
                assert list(m["labellers"]) == ["vlm", *VARIANTS, "stack"]


def test_bars_of_a_fresh_run(nan_run):
    """moments (2Q shards), loss initial, stack fit (T = EM = 10 steps), loss final (Q),
    predict and report (|F^te| frames), in this order, each from 0 to its total."""
    assert_progress(nan_run.err)
    assert screen(nan_run.err) == [
        Bar("moments", 8, 8, "shard"), Bar("loss initial", 4, 4, "shard"),
        Bar("stack fit", 10, 10, "step"), Bar("loss final", 4, 4, "shard"),
        Bar("predict", 4, 4, "frame"), Bar("report", 4, 4, "frame"),
    ]  # fmt: skip


def test_reads_of_a_fresh_run(nan_run):
    """x of each shard of S: twice in (M), once per pass of full_loss, once per epoch of
    (4); only the keys a step uses (never `c`, `m` of validation)."""
    for part, s in S_SHARDS:
        assert nan_run.reads.count(part, s, "x") == 2 + 1 + CONFIG["epochs"] + 1
    keys = {"train": {"ids", "y", "x"}, "validation": {"ids", "y", "x"},
            "test": {"ids", "y", "x", "g", "pred"}}  # fmt: skip
    assert all(key in keys[part] for part, _, key in nan_run.reads.log)


def test_run_json(nan_run):
    """(R): {out_dir}/run.json = {config (the mapping as written), sha256: {config, masks}},
    the SHA-256 of the config file and of {masks_dir}/run.json."""
    root = nan_run.out.parent
    config, masks = (
        (root / "gui-stack.yaml").read_bytes(),
        (root / "masks/run.json").read_bytes(),
    )
    sha = {
        k: hashlib.sha256(b).hexdigest()
        for k, b in (("config", config), ("masks", masks))
    }
    got = json.loads((nan_run.out / "run.json").read_text())
    assert got == {"config": yaml.safe_load(config), "sha256": sha}


def test_no_test_frame(tmp_path, repo):
    """Every test shard empty: `predict: nothing to do` (the empty prediction shards still
    written), `report: nothing to do`; agreement and macro recall null."""
    world = World(NAN_WORLD.columns, LABELS, {**NAN_WORLD.parts, "test": [EMPTY] * 3})
    run = run_main(tmp_path, repo, world, CONFIG)
    assert run.code == 0, run.err
    assert screen(run.err)[-2:] == ["predict: nothing to do", "report: nothing to do"]
    assert [f.read_text() for f in run.prediction_files()] == ["", "", ""]
    test = run.report()["test"]
    assert (test["agreement"], test["macro_recall"]) == (None, None)
