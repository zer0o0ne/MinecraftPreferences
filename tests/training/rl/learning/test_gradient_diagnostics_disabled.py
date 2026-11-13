"""Cheap RL events retain losses without full gradient or update copies."""

import json

import pytest
import torch

from mcpref.training.rl.learning.checkpoints import write_checkpoint
from training.pretrain.references.test_reference_resume import nested_equal
from training.rl.learning.loop_world import (
    checkpoints,
    install_steps,
    losses,
    model,
    run_train,
    write_unit_world,
    write_world,
)
from training.rl.learning.loss_fakes import install, modules


def forbid_gradient_diagnostics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled gradient diagnostics were invoked")

    for target in (
        "mcpref.training.rl.learning.loop.capture_gradients",
        "mcpref.training.rl.learning.producer._update_observation",
        "mcpref.training.rl.learning.producer.iter_update_terms",
        "mcpref.training.rl.learning.producer.on_update_observation",
    ):
        monkeypatch.setattr(target, forbidden)


def checkpoint_without_gradients(*args, **kwargs):
    assert all(
        p.grad is None for module in args[3].values() for p in module.parameters()
    )
    return write_checkpoint(*args, **kwargs)


def records(root):
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob("data/diagnostics/rl/*/metrics.json"))
    ]


def assert_same_training(actual, expected):
    assert nested_equal(model(actual), model(expected))
    assert losses(actual) == losses(expected)
    for path, other in zip(checkpoints(actual), checkpoints(expected), strict=True):
        assert nested_equal(
            torch.load(path, weights_only=True), torch.load(other, weights_only=True)
        )


def test_disabled_matches_enabled_state_and_preserves_output_events(
    tmp_path, repo, monkeypatch
):
    root, cfg, expected = write_world(
        tmp_path / "enabled", repo, gradient_diagnostics=True
    )
    install_steps(monkeypatch, [], all_parameters=True)
    run_train(root, cfg, expected, modules())
    enabled = records(root)
    root, cfg, actual = write_world(
        tmp_path / "disabled", repo, gradient_diagnostics=False
    )
    forbid_gradient_diagnostics(monkeypatch)
    monkeypatch.setattr(
        "mcpref.training.rl.learning.loop.write_checkpoint",
        checkpoint_without_gradients,
    )
    run_train(root, cfg, actual, modules())
    assert_same_training(actual, expected)
    disabled = records(root)
    assert len(disabled) == len(enabled) == 6
    for row, other in zip(disabled, enabled, strict=True):
        assert row["id"] == other["id"]
        assert row["joined"] == other["joined"]
        assert set(row["metrics"]) == {
            "update.learning_rate",
            "update.schedule_position",
            "update.logit_norm",
        }
        assert (
            row["metrics"]["update.logit_norm"] == other["metrics"]["update.logit_norm"]
        )
    state = json.loads((root / "data/diagnostics/run.json").read_text())
    assert state["channels"]["rl"]["complete"] is True


def test_disabled_recovers_checkpoint_and_pending_events(tmp_path, repo, monkeypatch):
    install_steps(monkeypatch, [], all_parameters=True)
    forbid_gradient_diagnostics(monkeypatch)
    root, cfg, expected = write_world(
        tmp_path / "whole", repo, gradient_diagnostics=False
    )
    run_train(root, cfg, expected, modules())
    expected_records = records(root)
    root, cfg, actual = write_world(tmp_path / "cut", repo, gradient_diagnostics=False)

    def interrupted(*args, **kwargs):
        checkpoint_without_gradients(*args, **kwargs)
        raise OSError("after checkpoint publication")

    with monkeypatch.context() as patch:
        patch.setattr("mcpref.training.rl.learning.loop.write_checkpoint", interrupted)
        with pytest.raises(OSError, match="after checkpoint publication"):
            run_train(root, cfg, actual, modules())
    run_train(root, cfg, actual, modules())
    assert_same_training(actual, expected)
    assert records(root) == expected_records
    assert not list(root.glob("data/diagnostics/outbox/rl/*.json"))


def test_real_backward_and_nonempty_ppo_shift_outputs_are_unchanged(
    tmp_path, repo, monkeypatch
):
    install(monkeypatch, [])
    root, cfg, expected = write_unit_world(
        tmp_path / "enabled", repo, epochs=2, gradient_diagnostics=True
    )
    run_train(root, cfg, expected, modules())
    enabled_records = records(root)
    root, cfg, actual = write_unit_world(
        tmp_path / "disabled", repo, epochs=2, gradient_diagnostics=False
    )
    forbid_gradient_diagnostics(monkeypatch)
    monkeypatch.setattr(
        "mcpref.training.rl.learning.loop.write_checkpoint",
        checkpoint_without_gradients,
    )
    run_train(root, cfg, actual, modules())
    assert_same_training(actual, expected)
    actual_records = records(root)
    assert len(actual_records) == len(enabled_records) == 6
    for row, other in zip(actual_records, enabled_records, strict=True):
        if row["id"].startswith("update:"):
            assert row["metrics"]["update.logit_norm"]["n"] == 1
            assert (
                row["metrics"]["update.logit_norm"]
                == other["metrics"]["update.logit_norm"]
            )
        else:
            assert row == other
