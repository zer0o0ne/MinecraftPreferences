"""Real policy artifacts: initial warm starts are separate from strict RL resumes."""

import json

import pytest
import torch
from models.policy.policy_oracle import cfg_of

from mcpref.models.policy.network import Policy
from mcpref.training.rl.cycle import load_model
from mcpref.training.rl.learning.checkpoints import restore
from mcpref.training.rl.learning.loop import load_initial_optimizer


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mode", ["classical", "continuous"])
@pytest.mark.parametrize("filename", ["model.pt", "checkpoint.pt"])
def test_initial_rl_loads_old_and_new_real_policy(tmp_path, legacy, mode, filename):
    source = Policy(cfg_of(seed=11), 2, ("a", "b"))
    target = Policy(cfg_of(seed=19), 2, ("a", "b"))
    expected_head = target.change_head.weight.detach().clone()
    weights = source.state_dict()
    if legacy:
        del weights["change_head.weight"]
    else:
        expected_head = weights["change_head.weight"]
    saved = {"pi": weights}
    if filename == "checkpoint.pt":
        saved = {"modules": saved}
    if mode == "continuous":
        saved["pretrain_mode"] = mode
    path = tmp_path / filename
    torch.save(saved, path)
    load_model(path, {"pi": target}, ("pi",), mode=mode, allow_legacy_policy=True)
    assert target.change_head.weight.equal(expected_head)
    for key, value in weights.items():
        assert target.state_dict()[key].equal(value)
    if legacy:
        with pytest.raises(RuntimeError, match="change_head.weight"):
            load_model(path, {"pi": target}, ("pi",), mode=mode)
    else:
        load_model(path, {"pi": target}, ("pi",), mode=mode)


@pytest.mark.parametrize("name", ["pi", "phi", "P"])
def test_warm_start_rejects_other_missing_weights(tmp_path, name):
    policy = Policy(cfg_of(), 2, ("a", "b"))
    modules = {"pi": policy, "phi": torch.nn.Linear(2, 2), "P": torch.nn.Linear(2, 1)}
    saved = {key: value.state_dict() for key, value in modules.items()}
    del saved["pi"]["change_head.weight"]
    del saved[name]["head.weight" if name == "pi" else "weight"]
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path, modules, ("phi", "pi", "P"), allow_legacy_policy=True)


def test_real_legacy_checkpoint_cannot_resume_weights_or_optimizer(tmp_path):
    source = Policy(cfg_of(), 2, ("a", "b"))
    del source.change_head
    target = Policy(cfg_of(), 2, ("a", "b"))
    other = {name: torch.nn.Linear(2, 2) for name in ("phi", "P", "metra")}
    saved = {name: module.state_dict() for name, module in other.items()}
    saved["pi"] = source.state_dict()
    old = torch.optim.AdamW(source.parameters(), lr=0.01)
    current = torch.optim.AdamW(target.parameters(), lr=0.01)
    directory = tmp_path / "checkpoints"
    shard = directory / "000000"
    shard.mkdir(parents=True)
    path = shard / "checkpoint.pt"
    torch.save({"modules": saved, "step": 1, "optimizer": old.state_dict()}, path)
    (tmp_path / "index").mkdir()
    (tmp_path / "index/run.json").write_text(json.dumps({"steps": 1}))
    with pytest.raises(RuntimeError, match="change_head.weight"):
        restore(directory, 1, other | {"pi": target})
    with pytest.raises(RuntimeError, match="mismatched params"):
        load_initial_optimizer(current, path)
