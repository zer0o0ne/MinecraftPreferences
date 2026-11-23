"""The public selector executes continuous index, trainer and diagnostic composition."""

import json
from dataclasses import asdict, replace

import pytest
import torch
import yaml
from training.pretrain.continuous.diagnostics.hooks_world import setup, training
from training.pretrain.continuous.diagnostics.test_hooks import assert_complete

from mcpref.runtime.sampling.checkpoints import load_weights
from mcpref.training.pretrain import modes
from mcpref.training.pretrain import run as dispatch
from mcpref.training.pretrain.continuous import loop
from mcpref.training.pretrain.continuous import run as continuous


@pytest.mark.parametrize("legacy_paths", [False, True])
def test_public_continuous_entrypoint_builds_index_and_completes_training(
    tmp_path, monkeypatch, legacy_paths
):
    world, initial_index = setup(tmp_path, monkeypatch)
    try:
        fixture = training(tmp_path, world, initial_index, "dispatch", gradients=True)
    finally:
        initial_index.close()
    for module in (modes, dispatch, continuous, loop):
        monkeypatch.setattr(module, "REPO", tmp_path)
    prefix = "configs/training/pretrain/"
    train_name = (
        prefix + "continuous/config.yaml" if legacy_paths else "dispatch-train.yaml"
    )
    index_name = (
        prefix + "continuous/index.yaml" if legacy_paths else "dispatch-index.yaml"
    )
    train_path = tmp_path / train_name
    train_path.parent.mkdir(parents=True, exist_ok=True)
    raw = vars(fixture.cfg) | {
        "index_dir": "dispatch-index",
        "out_dir": "dispatch",
        "retrain": False,
        "perception": "perception.yaml",
        "policy": "policy.yaml",
        "preference": "preference.yaml",
        "latent_init_scale": 1.0,
    }
    train_path.write_text(yaml.safe_dump(raw))
    (tmp_path / "perception.yaml").write_text(yaml.safe_dump(vars(world.phi_cfg)))
    (tmp_path / "policy.yaml").write_text("history: 3\n")
    (tmp_path / "mask.yaml").write_text("fixture: shared-action-catalogue\n")
    index_path = tmp_path / index_name
    index_path.write_text(
        yaml.safe_dump(
            asdict(
                replace(
                    world.cfg,
                    out_dir=raw["index_dir"],
                    training_config=train_name,
                )
            )
        )
    )
    selector = tmp_path / "pretrain.yaml"
    selector.write_text(
        yaml.safe_dump(
            {
                "mode": "continuous",
                "classical": {"config": "unused.yaml", "index": "unused-index.yaml"},
                "continuous": {
                    "config": prefix + "continuous.yaml"
                    if legacy_paths
                    else train_name,
                    "index": prefix + "continuous-index.yaml"
                    if legacy_paths
                    else index_name,
                },
            }
        )
    )
    constructed = []

    def build(perception, masks, policy, preference, paired, device, latent_scale):
        constructed.append(
            (perception, masks, policy, preference, paired, device, latent_scale)
        )
        return fixture.modules

    monkeypatch.setattr(continuous, "build_modules", build)
    monkeypatch.setattr(continuous, "perception_config", lambda _: world.phi_cfg)
    monkeypatch.setattr(continuous, "policy_config", lambda _: {"history": 3})
    monkeypatch.setattr(continuous, "preference_config", lambda _: {"history": 2})
    monkeypatch.setattr(continuous, "mask_config", lambda _: None)
    monkeypatch.setattr(continuous, "load_masks", lambda _: world.masks)
    assert not (tmp_path / "dispatch-index").exists()
    assert dispatch.main([str(selector)]) == 0
    assert len(constructed) == 1
    assert constructed[0][-3:] == (True, "cpu", 1.0)
    indexed = json.loads((tmp_path / "dispatch-index/run.json").read_text())
    assert indexed["complete"] and indexed["pretrain_mode"] == "continuous"
    assert "training_config" not in indexed["fingerprint"]
    run = json.loads((fixture.root / "run.json").read_text())
    assert run["complete"] and run["attention"] == "observation"
    saved = torch.load(fixture.root / "model.pt", weights_only=True)
    assert saved["pretrain_mode"] == "continuous" and saved["memory"] == "fragment"
    assert set(fixture.modules) <= saved.keys()
    checkpoint = next(fixture.root.glob("checkpoints/*/checkpoint.pt"))
    inference_weights = load_weights(checkpoint, "continuous")
    assert set(fixture.modules) <= inference_weights.keys()
    assert "stream" not in inference_weights
    assert_complete(tmp_path, "dispatch")
    assert (
        len(list(fixture.root.glob("checkpoints/*/gradient_flow/*/metrics.json"))) == 2
    )
