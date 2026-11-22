"""C4: RL checkpoint shards with CPU module, optimizer and torch RNG states."""

from pathlib import Path

import torch

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import (
    part_path,
    publish,
    shard_path,
    write_shard,
)
from mcpref.training.checkpoints import cpu_states, restore_rng

CHECKPOINT, LOSSES = "checkpoint.pt", "losses.json"
STATEFUL = ("phi", "pi", "P", "metra")


@measured("checkpoint")
def write_checkpoint(
    directory: Path,
    j: int,
    position: dict,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    losses: list[dict],
    *,
    mode: str = "classical",
) -> None:
    """Atomically publish one exact schedule position and its new loss records."""
    stateful = {name: modules[name] for name in STATEFUL}
    saved = position | {"modules": cpu_states(stateful)}
    saved |= {"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()}
    saved |= {
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    }
    if mode == "continuous":
        saved["pretrain_mode"] = mode
    with write_shard(directory, j) as staged:
        torch.save(saved, staged / CHECKPOINT)
        (staged / LOSSES).write_bytes(json_bytes(losses))


def restore(
    directory: Path,
    h: int,
    modules: dict[str, torch.nn.Module],
    *,
    mode: str = "classical",
) -> dict:
    """Restore checkpoint h - 1 weights and torch RNG on the modules' device."""
    state = torch.load(
        shard_path(directory, h - 1) / CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    check_mode(state, mode)
    for name in STATEFUL:
        modules[name].load_state_dict(state["modules"][name])
    device = next(modules["phi"].parameters()).device
    restore_rng(state, device)
    return state


def check_mode(state: dict, mode: str) -> None:
    actual = state.get("pretrain_mode", "classical")
    if actual != mode or mode not in ("classical", "continuous"):
        raise RuntimeError(
            f"RL checkpoint mode mismatch: artifact={actual}, run={mode}"
        )


def write_model(path: Path, modules: dict, *, mode: str = "classical") -> None:
    """Publish final weights, retaining the historical classical payload format."""
    saved = cpu_states(modules)
    if mode == "continuous":
        saved["pretrain_mode"] = mode
    part = part_path(path)
    torch.save(saved, part)
    publish(part, path)
