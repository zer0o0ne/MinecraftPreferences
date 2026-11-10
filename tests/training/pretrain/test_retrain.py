"""Frozen-phi heads retraining, separate-process loading and exact resume."""

import json

import torch

from training.pretrain.data.index_run import Interrupted, interrupting
from training.pretrain.loop_run import Loop
from training.pretrain.test_loop_resume import equal_states


def checkpoint(loop: Loop):
    paths = sorted((loop.out / "retrain/checkpoints").glob("*/checkpoint.pt"))
    return torch.load(paths[-1], map_location="cpu", weights_only=True)


def test_retrain_freezes_joint_phi_and_uses_only_head_optimizer(tmp_path):
    joint = Loop(tmp_path / "joint", keep_checkpoints=20).write()
    joint.run()
    initial = joint.model()
    full = Loop(tmp_path / "full", retrain=True, keep_checkpoints=20).write()
    full.run()
    final = full.model()
    assert equal_states({"phi": final["phi"]}, {"phi": initial["phi"]})
    assert final["phi_loss"]["lam"].equal(initial["phi_loss"]["lam"])
    assert not final["pi"]["Wa"].equal(initial["pi"]["Wa"])
    assert not final["P"]["u"].equal(initial["P"]["u"])
    saved = checkpoint(full)
    assert saved["step"] == 10
    assert len(saved["optimizer"]["param_groups"]) == 2
    assert [group["lr"] for group in saved["optimizer"]["param_groups"]] == [
        full.config["groups"]["pi"]["lr"] * full.config["lr_floor"],
        full.config["groups"]["pref"]["lr"] * full.config["lr_floor"],
    ]
    assert json.loads((full.out / "retrain/run.json").read_text()) == {
        "complete": True,
        "steps": 10,
    }


def test_interrupted_retrain_restores_heads_optimizer_rng_and_schedule(tmp_path):
    whole = Loop(tmp_path / "whole", retrain=True, keep_checkpoints=20).write()
    whole.run()
    cut = Loop(tmp_path / "cut", retrain=True, keep_checkpoints=20).write()
    cut.run(patch=interrupting("pretrain retrain", 4), raises=Interrupted)
    assert not (cut.out / "retrain/run.json").exists()
    torch.manual_seed(1923)
    torch.rand(5)
    cut.run()
    assert equal_states(cut.model(), whole.model())
    for j in range(4):
        a = cut.out / f"retrain/checkpoints/{j:06d}"
        b = whole.out / f"retrain/checkpoints/{j:06d}"
        assert (a / "losses.json").read_bytes() == (b / "losses.json").read_bytes()
        assert (a / "diagnostics.json").read_bytes() == (
            b / "diagnostics.json"
        ).read_bytes()
    assert (
        checkpoint(cut)["optimizer"]["state"].keys()
        == checkpoint(whole)["optimizer"]["state"].keys()
    )
    assert checkpoint(cut)["torch_rng"].equal(checkpoint(whole)["torch_rng"])


def test_completed_marker_skips_retrain_without_replacing_model(tmp_path):
    full = Loop(tmp_path / "full", retrain=True).write()
    full.run()
    before = (full.out / "model.pt").read_bytes()
    full.run()
    assert (full.out / "model.pt").read_bytes() == before


def test_new_process_between_joint_and_retrain_loads_joint_model(tmp_path):
    whole = Loop(tmp_path / "whole", retrain=True).write()
    whole.run()
    cut = Loop(tmp_path / "cut", retrain=True).write()

    def stop_before_retrain(mp):
        def stop(*args):
            raise Interrupted("before retrain")

        mp.setattr("mcpref.training.pretrain.retrain.retrain", stop)

    cut.run(patch=stop_before_retrain, raises=Interrupted)
    assert (cut.out / "model.pt").exists()
    assert not (cut.out / "retrain/checkpoints").exists()
    cut.run()
    assert equal_states(cut.model(), whole.model())
