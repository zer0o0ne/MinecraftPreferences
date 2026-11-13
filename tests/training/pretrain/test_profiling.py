"""Profiling a complete iteration preserves checkpoints, including retraining."""

import json

import pytest

from training.pretrain.test_diagnostics_joins import small_loop
from training.pretrain.test_gradient_diagnostics_disabled import assert_same_checkpoints


@pytest.mark.parametrize("retrain", [False, True])
def test_pretrain_profile_preserves_weights_optimizer_and_rng(
    tmp_path, monkeypatch, retrain
):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    expected = small_loop(
        tmp_path / "expected", gradient_diagnostics=False, retrain=retrain
    )
    expected.run()
    actual = small_loop(
        tmp_path / "actual", gradient_diagnostics=False, retrain=retrain
    )
    path = tmp_path / "profile.yaml"
    path.write_text("skip_steps: 1\nsteps: 1\ntop_ops: 5\n")
    monkeypatch.setenv("MCPREF_PROFILE", str(path))
    result = actual.run()
    assert result.out == ""
    assert_same_checkpoints(actual, expected)
    summaries = list(actual.out.glob("profiles/*/*/*/summary.json"))
    assert len(summaries) == 1 + retrain
    for path in summaries:
        row = json.loads(path.read_text())
        names = {region["name"] for region in row["regions"]}
        assert row["step"] == 2 and row["executed_step"] == 2
        assert {
            "step",
            "data.input_wait",
            "micro.forward",
            "backward",
            "optimizer",
            "checkpoint",
            "diagnostics.record",
            "diagnostics.reduce",
        } <= names
