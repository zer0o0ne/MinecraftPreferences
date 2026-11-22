"""Pretraining checkpoint shards with losses and diagnostics."""

import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import torch

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import shard_path, write_shard
from mcpref.models.policy.targets import ALIGNMENT
from mcpref.training.checkpoints import CHECKPOINT, Modules, cpu_states

if TYPE_CHECKING:
    from mcpref.training.pretrain.classical.gradients.flow import GradientStepRecord

LOSSES, DIAGNOSTICS = "losses.json", "diagnostics.json"


def check_alignment(path: Path) -> None:
    """Reject legacy training state before an existing result can be reused."""
    if path.exists():
        valid = json.loads(path.read_text()).get("policy_alignment") == ALIGNMENT
    else:
        valid = not any(
            (path.parent / name).exists()
            for name in ("model.pt", "joint-model.pt", "checkpoints", "retrain")
        )
    if not valid:
        raise RuntimeError(
            "Policy alignment changed; use a new training output directory"
        )


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
    saved = position | {"policy_alignment": ALIGNMENT, "modules": cpu_states(modules)}
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
    if state.get("policy_alignment") != ALIGNMENT:
        raise ValueError(
            "Checkpoint policy alignment differs; start a new training run"
        )
    for name, module in modules.items():
        module.load_state_dict(state["modules"][name])
    return state
