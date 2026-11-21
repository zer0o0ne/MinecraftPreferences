"""Atomic model, stream, archive-reference and RNG checkpoints."""

import torch

from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import (
    complete_shards,
    part_path,
    publish,
    shard_path,
    write_shard,
)
from mcpref.training.checkpoints import cpu_states, prune, restore_rng
from mcpref.training.pretrain.continuous.events import notify


def save(context, position, stream, archive, results, hooks) -> None:
    archive_state = archive.snapshot()
    device = torch.device(context.cfg.device)
    state = {
        "pretrain_mode": "continuous",
        "modules": cpu_states(context.modules),
        "optimizer": context.optimizer.state_dict(),
        "position": position,
        "stream": stream.detach("cpu"),
        "archive": archive_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }
    directory = context.root / "checkpoints"
    k = complete_shards(directory)
    with write_shard(directory, k) as staged:
        torch.save(state, staged / "checkpoint.pt")
        (staged / "metadata.json").write_bytes(
            json_bytes(
                {
                    "pretrain_mode": "continuous",
                    "attention": "observation",
                    "memory": "fragment",
                }
            )
        )
        records = [result.record(step) for step, result in results]
        (staged / "losses.json").write_bytes(json_bytes(records))
        notify(hooks, "checkpoint", context, staged, [result for _, result in results])
    notify(hooks, "published", context, k)
    prune(directory, k, context.cfg.keep_checkpoints)
    archive.prune_metadata(context.cfg.keep_checkpoints)


def restore(context):
    directory = context.root / "checkpoints"
    count = complete_shards(directory)
    if not count:
        return None
    # Own local checkpoints contain the explicitly versioned stream dataclasses.
    state = torch.load(
        shard_path(directory, count - 1) / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if state.get("pretrain_mode") != "continuous":
        raise ValueError("Cannot resume a classical checkpoint in continuous mode")
    for name, module in context.modules.items():
        module.load_state_dict(state["modules"][name])
    context.optimizer.load_state_dict(state["optimizer"])
    restore_rng(state, torch.device(context.cfg.device))
    state["stream"] = state["stream"].detach(context.cfg.device)
    return state


def write_model(context) -> None:
    path = context.root / "model.pt"
    staged = part_path(path)
    torch.save(
        {
            "pretrain_mode": "continuous",
            "attention": "observation",
            "memory": "fragment",
            **cpu_states(context.modules),
        },
        staged,
    )
    publish(staged, path)
