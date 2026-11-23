"""Index compatibility and the final retrained model are observable run contracts."""

import json
from dataclasses import make_dataclass

import pytest
import torch
from training.pretrain.continuous.engine_world import (
    Index,
    Reader,
    config,
    modules,
    observation_step,
    observations,
)
from training.pretrain.continuous.test_resume import world

from mcpref.infrastructure.shards.store import write_shard
from mcpref.models.policy.targets import ALIGNMENT
from mcpref.training.pretrain.continuous import loop, steps


def files(tmp_path, monkeypatch, **changes):
    monkeypatch.setattr(loop, "REPO", tmp_path)
    monkeypatch.setenv("MCPREF_PROFILE", "")
    paths = {
        key: f"{key}.yaml"
        for key in ("perception", "action_mask", "policy", "preference", "diagnostics")
    }
    for key, name in paths.items():
        (tmp_path / name).write_text(
            "history: 2\n" if key == "preference" else f"value: {key}\n"
        )
    (tmp_path / "index").mkdir()
    (tmp_path / "index/run.json").write_text("{}")
    path = tmp_path / "train.yaml"
    path.write_text("fixture: continuous\n")
    cfg = config(
        epochs=1,
        tbptt_states=2,
        index_dir="index",
        out_dir="run",
        retrain=True,
        **paths,
        **changes,
    )
    data = world(2)
    data.validate = lambda: None
    data.run["mathematics"] = {
        "tbptt_states": 2,
        "preference_history": 2,
        "action_mask": paths["action_mask"],
    }
    return cfg, path, data


@pytest.mark.parametrize(
    "field,value,message",
    [("preference_history", 3, "history differs")],
)
def test_incompatible_index_rejected_before_training(
    tmp_path, monkeypatch, field, value, message
):
    cfg, path, data = files(tmp_path, monkeypatch)
    data.run["mathematics"][field] = value
    with pytest.raises(ValueError, match=message):
        loop.pretrain(cfg, path, modules(), data)
    assert not (tmp_path / "run").exists()


def test_final_model_is_retrained_and_completed_rerun_is_stable(tmp_path, monkeypatch):
    cfg, path, data = files(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    loop.pretrain(cfg, path, modules(), data)
    root = tmp_path / "run"
    final = torch.load(root / "model.pt", weights_only=True)
    joint = torch.load(root / "joint-model.pt", weights_only=True)
    retrained = torch.load(root / "retrain/model.pt", weights_only=True)
    assert final["pretrain_mode"] == "continuous"
    for key, value in final["phi"].items():
        torch.testing.assert_close(value, joint["phi"][key], rtol=0, atol=0)
    assert any(not torch.equal(v, joint["pi"][key]) for key, v in final["pi"].items())
    for key, value in final["pi"].items():
        torch.testing.assert_close(value, retrained["pi"][key], rtol=0, atol=0)
    loop.pretrain(cfg, path, modules(), data)
    again = torch.load(root / "model.pt", weights_only=True)
    for key, value in final["pi"].items():
        torch.testing.assert_close(value, again["pi"][key], rtol=0, atol=0)
    assert json.loads((root / "run.json").read_text())["complete"]


def test_absent_preference_population_omits_only_its_known_groups(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MCPREF_PROFILE", "")
    monkeypatch.setattr(loop, "StreamReader", Reader)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    model = modules()
    del model["P"], model["P_loss"]
    raw = vars(config(epochs=1, tbptt_states=1))
    raw["groups"]["P_no_decay"] = {"lr": 0.01, "weight_decay": 0.0}
    cls = make_dataclass("Config", [(name, object) for name in raw], frozen=True)
    cfg = cls(**raw)
    data = Index(
        {"f": observations("x", [1])}, [], {"N_phi": 0, "N_pi": 0, "N_P": 0, "M": 1}
    )
    loop.phase(cfg, model, data, tmp_path / "valid", "joint", None)
    raw["groups"]["unknown"] = {"lr": 0.01, "weight_decay": 0.0}
    with pytest.raises(RuntimeError, match="unknown"):
        loop.phase(cls(**raw), model, data, tmp_path / "invalid", "joint", None)


@pytest.mark.parametrize("artifact", ["run", "model", "checkpoint"])
@pytest.mark.parametrize("semantics", [None, "disjoint-windows-v1"])
def test_legacy_training_artifacts_reject_resume(
    tmp_path, monkeypatch, artifact, semantics
):
    cfg, path, data = files(tmp_path, monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    value = {"pretrain_mode": "continuous", "policy_alignment": ALIGNMENT}
    if semantics is not None:
        value["gradient_semantics"] = semantics
    if artifact == "run":
        (root / "run.json").write_text(json.dumps(value))
    elif artifact == "model":
        torch.save(value, root / "model.pt")
    else:
        with write_shard(root / "checkpoints", 0) as staged:
            torch.save(value, staged / "checkpoint.pt")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="gradient semantics differ"):
        if artifact == "run":
            loop.pretrain(cfg, path, modules(), data)
        else:
            monkeypatch.setattr(loop, "StreamReader", Reader)
            loop.phase(cfg, modules(), data, root, "joint", None)
    assert all(p.read_bytes() == content for p, content in before.items())
