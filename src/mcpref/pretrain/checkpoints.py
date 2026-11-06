"""Interface Checkpoints, (T) 1, 2, 4 of specs/pretrain.md: checkpoint shards written,
restored and pruned; model.pt."""

import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import torch

from mcpref.runs.files import json_bytes
from mcpref.shards.store import part_path, publish, shard_path, sync, write_shard
from mcpref.training.devices import cpu_copies

if TYPE_CHECKING:
    from mcpref.pretrain.gradient_flow import GradientStepRecord

CHECKPOINT, LOSSES, DIAGNOSTICS = "checkpoint.pt", "losses.json", "diagnostics.json"
Modules = dict[str, torch.nn.Module]


def cpu_states(modules: Modules) -> dict[str, dict[str, torch.Tensor]]:
    """{module name: CPU copy of its state_dict()}, in the order of `modules`."""
    states = {name: module.state_dict() for name, module in modules.items()}
    return {
        name: dict(zip(s, cpu_copies(list(s.values())))) for name, s in states.items()
    }


def write_checkpoint(
    directory: Path,
    j: int,
    position: dict,
    modules: Modules,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    losses: list[dict],
    diagnostics: list[dict],
    gradient_flow: Sequence["GradientStepRecord"],
) -> None:
    """Write the unchanged optimizer state and all accepted-candidate diagnostics."""
    saved = position | {"modules": cpu_states(modules)}
    saved |= {"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()}
    cuda = device.type == "cuda"
    saved |= {"cuda_rng": torch.cuda.get_rng_state(device) if cuda else None}
    with write_shard(directory, j) as staged:
        torch.save(saved, staged / CHECKPOINT)
        (staged / LOSSES).write_bytes(json_bytes(losses))
        (staged / DIAGNOSTICS).write_bytes(json_bytes(diagnostics))
        for record in gradient_flow:
            name = PurePosixPath(record.metrics["evidence"]["pairs"]).parent.name
            target = staged / "gradient_flow" / name
            target.mkdir(parents=True)
            (target / "metrics.json").write_bytes(json_bytes(record.metrics))
            (target / "pairs.json").write_bytes(json_bytes(record.pairs))


def restore(directory: Path, h: int, modules: Modules) -> dict:
    """(T) 2: checkpoint.pt of checkpoint h - 1 loaded to the CPU; each module's
    load_state_dict."""
    state = torch.load(shard_path(directory, h - 1) / CHECKPOINT, map_location="cpu")
    for name, module in modules.items():
        module.load_state_dict(state["modules"][name])
    return state


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
