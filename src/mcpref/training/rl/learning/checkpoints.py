"""C4: RL checkpoint shards with CPU module, optimizer and torch RNG states."""

from pathlib import Path

import torch

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import shard_path, write_shard
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
) -> None:
    """Atomically publish one exact schedule position and its new loss records."""
    stateful = {name: modules[name] for name in STATEFUL}
    saved = position | {"modules": cpu_states(stateful)}
    saved |= {"optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()}
    saved |= {
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    }
    with write_shard(directory, j) as staged:
        torch.save(saved, staged / CHECKPOINT)
        (staged / LOSSES).write_bytes(json_bytes(losses))


def restore(directory: Path, h: int, modules: dict[str, torch.nn.Module]) -> dict:
    """Restore checkpoint h - 1 weights and torch RNG on the modules' device."""
    state = torch.load(
        shard_path(directory, h - 1) / CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    for name in STATEFUL:
        modules[name].load_state_dict(state["modules"][name])
    device = next(modules["phi"].parameters()).device
    restore_rng(state, device)
    return state
