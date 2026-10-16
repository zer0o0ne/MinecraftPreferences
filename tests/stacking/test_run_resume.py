"""`main` of src/mcpref/stacking/run.py rerun against specs/gui-stack.md (Behaviour (F): the
`model.pt` skip, the continued fit; (P): continued after the complete prediction shards;
progress bars of a continued and a finished run) on NAN_WORLD of stack_world.py."""

import math

import numpy as np
import pytest
import torch
from progress_bars import Bar, assert_aborted, screen, states

from . import fit_oracle as fo
from . import stack_oracle as so
from .stack_files import run, run_main
from .stack_world import CONFIG, LABELS, NAN_WORLD, reference

S_SHARDS = [("train", 0), ("train", 2), ("validation", 0), ("validation", 1)]
Q, T = 4, 10
DONE = [
    "loss initial: already done",
    "stack fit: already done",
    "loss final: already done",
]
MOMENTS, REPORT = Bar("moments", 8, 8, "shard"), Bar("report", 4, 4, "frame")


def outputs(out) -> dict:
    """The bytes of model.pt and of the prediction shards."""
    files = [
        out / "model.pt",
        *sorted((out / "predictions").glob("*/predictions.jsonl")),
    ]
    return {f.relative_to(out): f.read_bytes() for f in files}


def test_finished_run(tmp_path, repo):
    """A rerun after a complete run: only (M) reads x of S (each shard twice); fit, the
    losses and the predictions `already done`; the same outputs; `report` redrawn."""
    first = run_main(tmp_path, repo, NAN_WORLD, CONFIG)
    before = outputs(first.out)
    again = run(tmp_path, repo, CONFIG)
    assert again.code == 0, again.err
    assert screen(again.err) == [MOMENTS, *DONE, "predict: already done", REPORT]
    assert [again.reads.count(p, s, "x") for p, s in S_SHARDS] == [2] * Q
    assert not [e for e in again.reads.log if e[0] == "test" and e[2] == "x"]
    assert outputs(again.out) == before and again.report() == first.report()


def test_model_file_skips_fit(tmp_path, repo):
    """model.pt present (other theta_T and losses written into it; checkpoints and
    predictions deleted): fit skipped, theta_T and both losses read from it."""
    first = run_main(tmp_path, repo, NAN_WORLD, CONFIG)
    model = first.model()
    model["params"] = [-2 * p for p in model["params"]]
    model |= {"loss_initial": 1.25, "loss_final": 2.5}
    torch.save(model, first.out / "model.pt")
    for d in ("checkpoints", "predictions"):
        for f in sorted((first.out / d).glob("**/*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    again = run(tmp_path, repo, CONFIG)
    assert again.code == 0, again.err
    assert screen(again.err) == [MOMENTS, *DONE, Bar("predict", 4, 4, "frame"), REPORT]
    train = again.report()["train"]
    assert (train["loss_initial"], train["loss_final"]) == (1.25, 2.5)
    theta = [p.double().numpy() for p in model["params"]]
    z = reference(NAN_WORLD, CONFIG)["z_te"]
    for records, zr in zip(again.predictions(), z):
        want = np.exp(so.log_softmax(so.logits(theta, zr)))
        got = np.array([[r["p"][k] for k in LABELS] for r in records]).reshape(
            want.shape
        )
        assert np.allclose(got, want, rtol=1e-5, atol=1e-8)


def fit_reads(log: list) -> int:
    """The x reads of shards of S so far."""
    return sum(1 for p, _, key in log if key == "x" and p != "test")


def last_checkpoint(visits_done: int) -> int:
    """The step of the last checkpoint after the first `visits_done` visits of (4)."""
    sizes = reference(NAN_WORLD, CONFIG)["sizes"]
    rng, visits = np.random.Generator(np.random.PCG64(CONFIG["seed"])), []
    for _ in range(CONFIG["epochs"]):
        visits += dict.fromkeys(q for q, _ in fo.epoch(rng, sizes, CONFIG["batch"]))
    steps = sum(math.ceil(sizes[q] / CONFIG["batch"]) for q in visits[:visits_done])
    return steps // CONFIG["checkpoint_steps"] * CONFIG["checkpoint_steps"]


def test_fit_interrupted_then_continued(tmp_path, repo, nan_run):
    """The read of the 7th visit of (4) fails (after 2Q reads of (M), Q of L(theta_0)); the
    rerun restores the last checkpoint: `loss initial: already done`, `stack fit` drawn from
    its step; theta_T, the losses and the predictions those of an uninterrupted run."""
    t = last_checkpoint(6)
    assert 0 < t < T
    cut = run_main(
        tmp_path, repo, NAN_WORLD, CONFIG, fail=lambda log: fit_reads(log) == 19
    )
    assert_aborted(cut.code, cut.err)
    again = run(tmp_path, repo, CONFIG)
    assert again.code == 0, again.err
    assert screen(again.err) == [MOMENTS, DONE[0], Bar("stack fit", T, T, "step"),
                                 Bar("loss final", Q, Q, "shard"),
                                 Bar("predict", 4, 4, "frame"), REPORT]  # fmt: skip
    assert states(again.err, "stack fit")[0] == Bar("stack fit", t, T, "step")
    got, want = again.model()["params"], nan_run.model()["params"]
    assert all(torch.equal(p, q) for p, q in zip(got, want, strict=True))
    assert outputs(again.out) == outputs(nan_run.out)
    assert again.report()["train"] == nan_run.report()["train"]


@pytest.mark.parametrize("broken", [1, 2])
def test_predictions_interrupted_then_continued(tmp_path, repo, nan_run, broken):
    """The read of x of test shard `broken` fails: the prediction shards before it are
    complete; the rerun reads x of the remaining test shards only, its `predict` bar starts
    at their frames, and the prediction shards equal those of an uninterrupted run."""
    fail = lambda log: log[-1] == ("test", broken, "x")
    cut = run_main(tmp_path, repo, NAN_WORLD, CONFIG, fail=fail)
    assert_aborted(cut.code, cut.err)
    names = sorted(
        p.name for p in (cut.out / "predictions").iterdir() if p.suffix != ".part"
    )
    assert names == [f"{r:06d}" for r in range(broken)]
    again = run(tmp_path, repo, CONFIG)
    assert again.code == 0, again.err
    assert screen(again.err) == [MOMENTS, *DONE, Bar("predict", 4, 4, "frame"), REPORT]
    assert states(again.err, "predict")[0] == Bar("predict", 2, 4, "frame")
    read = [s for p, s, key in again.reads.log if (p, key) == ("test", "x")]
    assert read == list(range(broken, 3))
    assert outputs(again.out) == outputs(nan_run.out)
