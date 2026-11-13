"""RL profiles capture real backward and complete joint/retrain iterations."""

import json

import pytest

from mcpref.training.rl.learning.retrain import retrain as run_retrain
from training.pretrain.data.index_run import repo_root
from training.rl.learning.loop_world import run_train, write_unit_world
from training.rl.learning.loss_fakes import history, install, modules
from training.rl.learning.test_gradient_diagnostics_disabled import assert_same_training


@pytest.mark.parametrize("retrain", [False, True])
def test_rl_profile_preserves_training(tmp_path, repo, monkeypatch, retrain):
    install(monkeypatch, [])
    monkeypatch.setattr(
        "mcpref.training.rl.learning.retrain.encode_ref",
        lambda ref, phi, *args: history(ref.index, phi),
    )
    monkeypatch.setenv("MCPREF_PROFILE", "")
    root, cfg, expected = write_unit_world(
        tmp_path / "expected", repo, epochs=2, gradient_diagnostics=False
    )
    mods = modules()
    run_train(root, cfg, expected, mods)
    if retrain:
        with repo_root(root):
            run_retrain(cfg, mods, expected, None)
    root, cfg, actual = write_unit_world(
        tmp_path / "actual", repo, epochs=2, gradient_diagnostics=False
    )
    path = tmp_path / "profile.yaml"
    path.write_text("skip_steps: 1\nsteps: 1\ntop_ops: 5\n")
    monkeypatch.setenv("MCPREF_PROFILE", str(path))
    mods = modules()
    run_train(root, cfg, actual, mods)
    if retrain:
        with repo_root(root):
            run_retrain(cfg, mods, actual, None)
    assert_same_training(actual, expected)
    summaries = list(actual.parent.glob("profiles/*/*/*/summary.json"))
    assert len(summaries) == 1 + retrain
    for path in summaries:
        row = json.loads(path.read_text())
        names = {region["name"] for region in row["regions"]}
        assert row["step"] == 2 and row["executed_step"] == 2
        assert {
            "step",
            "rl.gradient_histories",
            "rl.reward_histories",
            "rl.heads_losses",
            "backward",
            "optimizer",
            "checkpoint",
        } <= names
