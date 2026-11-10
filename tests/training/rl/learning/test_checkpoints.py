"""RL checkpoint serialization: CPU modules, optimizer, position, losses, and torch RNG."""

import json

import numpy as np
import torch

from mcpref.training.rl.learning.checkpoints import restore, write_checkpoint
from training.rl.learning.loss_fakes import modules


def optimizer(mods):
    return torch.optim.AdamW(
        [
            parameter
            for name in ["phi", "pi", "P", "metra"]
            for parameter in mods[name].parameters()
        ],
        lr=0.01,
        betas=(0.8, 0.9),
        eps=1.0e-9,
        weight_decay=0.02,
    )


def test_checkpoint_round_trip_restores_every_rl_component(tmp_path):
    mods = modules()
    opt = optimizer(mods)
    mods["phi"].scale.grad = torch.tensor(0.5, dtype=torch.float64)
    opt.step()
    opt.zero_grad(set_to_none=True)
    position = {"step": 3, "epoch": 2,
                "schedule_rng": np.random.Generator(np.random.PCG64(8)).bit_generator.state,
                "batch_in_epoch": 1}  # fmt: skip
    logged = [{"step": 3, "ppo": 1.0, "shift": None, "metra": 2.0, "P": None,
               "loss": 3.0, "lr": {"phi_head": 0.001}}]  # fmt: skip
    torch.manual_seed(41)
    write_checkpoint(tmp_path, 0, position, mods, opt, torch.device("cpu"), logged)
    expected_random = torch.rand(3)
    shard = tmp_path / "000000"
    raw = torch.load(shard / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert set(raw) == {"step", "epoch", "schedule_rng", "batch_in_epoch", "modules",
                        "optimizer", "torch_rng", "cuda_rng"}  # fmt: skip
    assert set(raw["modules"]) == {"phi", "pi", "P", "metra"}
    assert raw["cuda_rng"] is None
    assert json.loads((shard / "losses.json").read_text()) == logged
    with torch.no_grad():
        for module in mods.values():
            for parameter in module.parameters():
                parameter.add_(100)
    torch.manual_seed(999)
    result = restore(tmp_path, 1, mods)
    assert {key: result[key] for key in position} == position
    assert set(result["optimizer"]) == {"state", "param_groups"}
    for name in ["phi", "pi", "P", "metra"]:
        for key, value in raw["modules"][name].items():
            assert value.device.type == "cpu"
            assert value.equal(mods[name].state_dict()[key])
    restored_optimizer = optimizer(mods)
    restored_optimizer.load_state_dict(result["optimizer"])
    assert (
        restored_optimizer.state_dict()["param_groups"]
        == opt.state_dict()["param_groups"]
    )
    assert torch.rand(3).equal(expected_random)
