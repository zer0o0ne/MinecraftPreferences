"""Sliding K-observation graphs from an evolving detached memory anchor."""

from contextlib import contextmanager

import torch

from mcpref.training.pretrain.continuous.state import detached


def capture_rng(device):
    return {
        "cpu": bytes(torch.get_rng_state().tolist()),
        "cuda": bytes(torch.cuda.get_rng_state(device).tolist())
        if device.type == "cuda"
        else None,
    }


@contextmanager
def replay_rng(state, device):
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.set_rng_state(
            torch.frombuffer(bytearray(state["cpu"]), dtype=torch.uint8)
        )
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(
                torch.frombuffer(bytearray(state["cuda"]), dtype=torch.uint8), device
            )
        yield


def refresh(stream, latents):
    for key in stream.recent.keys() & latents.keys():
        stream.recent[key] = latents[key]
    for row in stream.policy:
        row["z"] = latents.get(row["id"], row["z"])
    if stream.previous is not None:
        row = stream.previous
        row["z"] = latents.get(row["id"], row["z"])
    if stream.pending is not None:
        pending = stream.pending
        pending["z"] = latents.get(pending["id"], pending["z"])
        pending["Y"] = torch.stack(
            [latents.get(key, z) for key, z in zip(pending["indices"], pending["Y"][0])]
        )[None]
        pending["readout"] = None


def observation(context, stream, current, forward):
    device = torch.device(context.cfg.device)
    stream.replay.append(
        {"id": current.id, "frame": current.frame, "line": current.line, "rng": None}
    )
    state, latents, first = stream.anchor, {}, None
    for i, entry in enumerate(stream.replay):
        args = context.modules["phi"], entry["frame"], entry["line"], state
        if entry["rng"] is None:
            entry["rng"] = capture_rng(device)
            z, state = forward(*args)
        else:
            with replay_rng(entry["rng"], device):
                z, state = forward(*args)
        latents[entry["id"]] = z
        if i == 0:
            first = detached(state)
    stream.recurrent = state
    refresh(stream, latents)
    if len(stream.replay) == context.cfg.tbptt_states:
        stream.anchor = first
        stream.replay.pop(0)
    return latents[current.id]
