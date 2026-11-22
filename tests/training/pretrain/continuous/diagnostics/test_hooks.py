"""Actual DiagnosticRun lifecycle preserves training and recovers producer outboxes."""

import json

import pytest
import torch
from training.pretrain.continuous.diagnostics.hooks_world import (
    accepted,
    equal_weights,
    setup,
    training,
    weights,
)

from mcpref.training.pretrain.continuous import loop
from mcpref.training.pretrain.continuous.diagnostics.hooks import Diagnostics


def assert_policy(record, source, count):
    assert record["metrics"]["policy.nll"]["n"] == count
    assert record["populations"]["policy.nll"] == [source] * count
    assert record["populations"]["policy.top1_support"] == [source] * count
    assert record["metrics"]["policy.top1_support"]["insufficient_clusters"]
    joined = record["joined"]["policy"]
    assert sum(map(sum, joined["confusion"])) == count
    assert sum(row["n"] for row in joined["reliability"]) == count


def assert_episode_populations(tmp_path, name, records):
    # The seed-7 order is f0 then f1; old:3 completes at step 3, the rest at step 5.
    clusters = ([], [], ["source-1"], [], ["source-0"] * 3)
    for i, expected in enumerate(clusters, 1):
        row = records[f"continuous:joint:step:{i}"]
        assert row["context"]["cluster_ids"] == expected
        for key in ("P", "P_inc", "P_succ", "pretrain.progress"):
            assert row["populations"][key] == expected
    selection = json.loads(
        (tmp_path / name / "reference/000000/selection.json").read_text()
    )["items"]
    assert any(edge["original_a"] == "c" for edge in selection)
    for record in records.values():
        if record["context"]["producer"] != "pretrain-continuous-reference":
            continue
        expected = [
            "source-1" if edge["original_a"] == "c" else "source-0"
            for edge in selection
            if edge["part"] == record["context"]["split"]
        ]
        assert record["context"]["cluster_ids"] == expected
        assert record["populations"]["P"] == expected
    assert_policy(records["continuous:joint:step:2"], "source-1", 1)
    assert_policy(records["continuous:joint:step:5"], "source-0", 2)


def assert_complete(tmp_path, name):
    root = tmp_path / f"{name}-diagnostics"
    run = json.loads((root / "run.json").read_text())
    step, reference = (
        run["channels"][f"pretrain-continuous-joint-{kind}"]
        for kind in ("step", "reference")
    )
    assert run["complete"] and step["complete"] and reference["complete"]
    assert (len(step["events"]), step["records"], step["next"]) == (5, 5, 6)
    assert [event["id"] for event in step["events"]] == [
        f"step:{i}" for i in range(1, 6)
    ]
    assert (len(reference["events"]), reference["records"], reference["next"]) == (
        3,
        24,
        3,
    )
    assert not list((root / "outbox").rglob("*.json"))
    result = accepted(root)
    assert len(result) == 29
    for i in range(1, 6):
        row = result[f"continuous:joint:step:{i}"]
        assert row["context"]["steps"] == [i]
        assert row["joined"]["reason"] is None
    assert_episode_populations(tmp_path, name, result)
    return result


@pytest.mark.parametrize("gradients", [False, True])
def test_diagnostic_hooks_leave_model_and_rng_unchanged(
    tmp_path, monkeypatch, gradients
):
    world, data = setup(tmp_path, monkeypatch)
    try:
        plain = training(tmp_path, world, data, "plain")
        torch.manual_seed(31)
        loop.train(plain)
        expected_rng = torch.get_rng_state()
        monitored = training(tmp_path, world, data, "monitored", gradients=gradients)
        torch.manual_seed(31)
        loop.train(monitored, Diagnostics())
        equal_weights(weights(plain.root), weights(monitored.root))
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_complete(tmp_path, "monitored")
        files = list(monitored.root.glob("checkpoints/*/gradient_flow/*/metrics.json"))
        assert bool(files) == gradients
    finally:
        data.close()


def test_diagnostic_hooks_without_preference_population(tmp_path, monkeypatch):
    world, data = setup(tmp_path, monkeypatch, preferences=False)
    try:
        assert data.run["populations"]["train"]["N_P"] == 0
        monitored = training(tmp_path, world, data, "own-only", gradients=True)
        assert "P" not in monitored.modules and "P_loss" not in monitored.modules
        diagnostics = Diagnostics()
        loop.train(monitored, diagnostics)
        assert diagnostics.reference.fit is None
        assert diagnostics.reference.initial is None
        assert not (monitored.root / "reference").exists()
        assert not list(monitored.root.glob("checkpoints/*/gradient_flow"))
        assert "P" not in weights(monitored.root)
        root = tmp_path / "own-only-diagnostics"
        run = json.loads((root / "run.json").read_text())
        reference = run["channels"]["pretrain-continuous-joint-reference"]
        assert run["complete"] and reference["complete"]
        assert (reference["records"], reference["events"], reference["next"]) == (
            0,
            [],
            0,
        )
        records = accepted(root)
        assert len(records) == 5
        for record in records.values():
            assert record["populations"]["P"] == []
            assert record["metrics"]["P"]["reason"] == "empty_pair_population"
        assert_policy(records["continuous:joint:step:2"], "source-1", 1)
        assert_policy(records["continuous:joint:step:5"], "source-0", 2)
        assert not list((root / "outbox").rglob("*.json"))
    finally:
        data.close()


class InterruptedDiagnostics(Diagnostics):
    def published(self, context, ordinal):
        if context.step == 2:
            raise InterruptedError(
                "checkpoint committed before diagnostics publication"
            )
        super().published(context, ordinal)


def test_recovery_after_checkpoint_commit_publishes_outbox_and_reference_once(
    tmp_path, monkeypatch
):
    world, data = setup(tmp_path, monkeypatch)
    try:
        full = training(tmp_path, world, data, "full", gradients=True)
        torch.manual_seed(31)
        loop.train(full, Diagnostics())
        expected_rng = torch.get_rng_state()
        interrupted = training(tmp_path, world, data, "resumed", gradients=True)
        torch.manual_seed(31)
        with pytest.raises(InterruptedError, match="checkpoint committed"):
            loop.train(interrupted, InterruptedDiagnostics())
        staged = list((tmp_path / "resumed-diagnostics/outbox").rglob("*.json"))
        assert len(staged) == 2
        assert (interrupted.root / "checkpoints/000000/checkpoint.pt").exists()
        resumed = training(tmp_path, world, data, "resumed", gradients=True)
        torch.manual_seed(938)
        loop.train(resumed, Diagnostics())
        equal_weights(weights(full.root), weights(resumed.root))
        assert torch.equal(torch.get_rng_state(), expected_rng)
        expected, actual = (
            assert_complete(tmp_path, "full"),
            assert_complete(tmp_path, "resumed"),
        )
        assert expected.keys() == actual.keys()
        for key in expected:
            for field in ("metrics", "joined", "populations"):
                assert expected[key][field] == actual[key][field]
        for name in ("losses.json", "diagnostics.json"):
            for path in full.root.glob(f"checkpoints/*/{name}"):
                other = resumed.root / path.relative_to(full.root)
                assert json.loads(path.read_text()) == json.loads(other.read_text())
    finally:
        data.close()
