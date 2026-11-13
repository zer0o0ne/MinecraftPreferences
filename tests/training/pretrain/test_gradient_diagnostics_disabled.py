"""Disabled gradient diagnostics preserve training and release gradients before I/O."""

import json

import pytest
import torch

from mcpref.training.pretrain.checkpoints import write_checkpoint
from training.pretrain.references.test_reference_resume import nested_equal
from training.pretrain.test_diagnostics_joins import records, small_loop


def forbid_gradient_diagnostics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled gradient diagnostics were invoked")

    for target in (
        "mcpref.training.pretrain.startup.select_gradient_steps",
        "mcpref.training.pretrain.steps.steps.selected_step",
        "mcpref.training.pretrain.steps.step.gradient_stats",
        "mcpref.training.pretrain.steps.step.update_stats",
        "torch.autograd.grad",
    ):
        monkeypatch.setattr(target, forbidden)


def checkpoint_without_gradients(*args, **kwargs):
    optimizer = args[4]
    assert all(
        p.grad is None for group in optimizer.param_groups for p in group["params"]
    )
    return write_checkpoint(*args, **kwargs)


def assert_same_checkpoints(actual, expected):
    paths = sorted(actual.out.glob("**/checkpoint.pt"))
    assert paths
    for path in paths:
        other = expected.out / path.relative_to(actual.out)
        assert nested_equal(
            torch.load(path, weights_only=True), torch.load(other, weights_only=True)
        )
    assert nested_equal(actual.model(), expected.model())


@pytest.mark.parametrize("retrain", [False, True])
def test_disabled_matches_enabled_weights_adam_rng_and_cheap_metrics(
    tmp_path, monkeypatch, retrain
):
    enabled = small_loop(
        tmp_path / "enabled", gradient_diagnostics=True, micro_batch=1, retrain=retrain
    )
    enabled.run()
    disabled = small_loop(
        tmp_path / "disabled",
        gradient_diagnostics=False,
        micro_batch=1,
        retrain=retrain,
    )
    forbid_gradient_diagnostics(monkeypatch)
    monkeypatch.setattr(
        "mcpref.training.pretrain.steps.steps.write_checkpoint",
        checkpoint_without_gradients,
    )
    monkeypatch.setattr(
        "mcpref.training.pretrain.retrain.write_checkpoint",
        checkpoint_without_gradients,
    )
    result = disabled.run()
    assert "pretrain gradient flow: disabled" in result.err
    assert not (disabled.out / "gradient-flow").exists()
    assert not list(disabled.out.glob("checkpoints/*/gradient_flow"))
    assert_same_checkpoints(disabled, enabled)
    for path in disabled.out.glob("**/losses.json"):
        actual = json.loads(path.read_text())
        expected = json.loads(
            (enabled.out / path.relative_to(disabled.out)).read_text()
        )
        for row, other in zip(actual, expected, strict=True):
            assert row["gradients"] is None and row["updates"] is None
            assert row == other | {"gradients": None, "updates": None}
    for path in disabled.out.glob("**/diagnostics.json"):
        for row in json.loads(path.read_text()):
            assert row["gradients"] is None and row["updates"] is None
            assert row["losses"] and row["preference"]
    assert records(disabled) == records(enabled)


def test_disabled_resume_after_published_checkpoint(tmp_path, monkeypatch):
    options = {
        "gradient_diagnostics": False,
        "diagnostic_window": 1,
        "checkpoint_steps": 1,
    }
    expected = small_loop(tmp_path / "expected", **options)
    forbid_gradient_diagnostics(monkeypatch)
    expected.run()
    actual = small_loop(tmp_path / "actual", **options)

    def interrupted(*args, **kwargs):
        checkpoint_without_gradients(*args, **kwargs)
        raise OSError("after checkpoint publication")

    with monkeypatch.context() as patch:
        patch.setattr(
            "mcpref.training.pretrain.steps.steps.write_checkpoint", interrupted
        )
        actual.run(raises=OSError)
    actual.run()
    assert_same_checkpoints(actual, expected)
    assert records(actual) == records(expected)
    assert not (actual.out / "gradient-flow").exists()
