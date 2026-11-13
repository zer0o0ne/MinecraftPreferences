"""Pretraining checkpoint shards with losses and diagnostics."""

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import torch

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import shard_path, write_shard
from mcpref.training.checkpoints import CHECKPOINT, Modules, cpu_states

if TYPE_CHECKING:
    from mcpref.training.pretrain.gradients.flow import GradientStepRecord

LOSSES, DIAGNOSTICS = "losses.json", "diagnostics.json"


@measured("checkpoint")
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
