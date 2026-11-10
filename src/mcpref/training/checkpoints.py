"""Model and RNG checkpoint operations shared by training loops."""

import os
from pathlib import Path

import torch

from mcpref.infrastructure.shards.store import part_path, publish, shard_path, sync
from mcpref.training.devices import cpu_copies

CHECKPOINT = "checkpoint.pt"
Modules = dict[str, torch.nn.Module]


def cpu_states(modules: Modules) -> dict[str, dict[str, torch.Tensor]]:
    """{module name: CPU copy of its state_dict()}, in the order of `modules`."""
    states = {name: module.state_dict() for name, module in modules.items()}
    return {
        name: dict(zip(s, cpu_copies(list(s.values())))) for name, s in states.items()
    }


def restore_rng(state: dict, device: torch.device) -> None:
    """(T) 2: torch's CPU generator, and on CUDA the device's, from the checkpoint."""
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def prune(directory: Path, last: int, k: int) -> None:
    """The checkpoint.pt of each checkpoint 0 .. last - k that still has one deleted
    (os.unlink, then sync of its directory)."""
    for j in range(last - k + 1):
        path = shard_path(directory, j) / CHECKPOINT
        if path.exists():
            os.unlink(path)
            sync(path.parent)


def write_model(path: Path, modules: Modules) -> None:
    """(T) 4: {module name: CPU copy of its state_dict()} by torch.save to <path>.part,
    then publish."""
    part = part_path(path)
    torch.save(cpu_states(modules), part)
    publish(part, path)
