"""Primitive trajectory records, continuation state and raw history of the spec."""

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from mcpref.env.state import EnvContinuation, EnvState
from mcpref.mcts.tree import LatentHistory, Tree
from mcpref.perception.phi import Phi
from mcpref.perception.stream import PhiStream
from mcpref.sampling.codecs import (
    _cpu,
    _decode_continuation,
    _decode_history,
    _decode_observation,
    _decode_tree,
    _encode_continuation,
    _encode_history,
    _encode_observation,
    _encode_tree,
)
from mcpref.sampling.models import decode_stream, encode_stream
from mcpref.shards.store import complete_shards, name, shard_path


@dataclass(frozen=True)
class StepRecord:
    """Interface: one delayed transition and its independent pre-advance tree."""

    index: int
    before: EnvState
    after: EnvState
    action: int
    probabilities: Tensor
    delta: Tensor
    predicted: Tensor
    elapsed_ms: float
    preference_close: dict[str, float]
    preference_start: dict[str, float]
    tree: dict


@dataclass
class SamplerState:
    """Interface: the complete state at a trajectory boundary."""

    observation: EnvState
    env: EnvContinuation
    history: LatentHistory
    tree: Tree
    rng_state: dict
    next_step: int
    stream: PhiStream
    fingerprint: str


@dataclass(frozen=True)
class RawHistory:
    """Interface: raw states and their preceding actions from a stream restart."""

    states: list[EnvState]
    actions: list[int | None]
    restart: int


def _record_mapping(record: StepRecord) -> dict:
    return {
        "index": record.index,
        "before": _encode_observation(record.before),
        "after": _encode_observation(record.after),
        "action": record.action,
        "probabilities": _cpu(record.probabilities),
        "delta": _cpu(record.delta),
        "predicted": _cpu(record.predicted),
        "elapsed_ms": record.elapsed_ms,
        "preference_close": record.preference_close,
        "preference_start": record.preference_start,
        "tree": record.tree,
    }


def _decode_record(value: dict) -> StepRecord:
    return StepRecord(
        value["index"],
        _decode_observation(value["before"]),
        _decode_observation(value["after"]),
        value["action"],
        value["probabilities"],
        value["delta"],
        value["predicted"],
        value["elapsed_ms"],
        value["preference_close"],
        value["preference_start"],
        value["tree"],
    )


def write_step(directory: Path, local: int, record: StepRecord) -> None:
    """Interface: steps/<name(local)>.pt in the staged trajectory shard."""
    steps = directory / "steps"
    steps.mkdir(exist_ok=True)
    torch.save(_record_mapping(record), steps / f"{name(local)}.pt")


def write_state(directory: Path, state: SamplerState) -> None:
    """Interface: state.pt after the final successor has been encoded and advanced."""
    value = {
        "observation": _encode_observation(state.observation),
        "env": _encode_continuation(state.env),
        "history": _encode_history(state.history),
        "tree": _encode_tree(state.tree),
        "rng": copy.deepcopy(state.rng_state),
        "next_step": state.next_step,
        "stream": encode_stream(state.stream),
        "fingerprint": state.fingerprint,
    }
    torch.save(value, directory / "state.pt")


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_state(
    out_dir: Path, phi: Phi, states: int, device: torch.device
) -> SamplerState:
    """Interface: decode the last complete shard's state, one shard only."""
    directory = shard_path(out_dir / "shards", complete_shards(out_dir / "shards") - 1)
    value = _load(directory / "state.pt")
    try:
        return SamplerState(
            _decode_observation(value["observation"]),
            _decode_continuation(value["env"]),
            _decode_history(value["history"], device),
            _decode_tree(value["tree"], device),
            value["rng"],
            value["next_step"],
            decode_stream(phi, states, value["stream"], device),
            value["fingerprint"],
        )
    except KeyError:
        raise RuntimeError("incompatible sampling state") from None


def _steps_per_shard(out_dir: Path) -> int:
    return _load(shard_path(out_dir / "shards", 0) / "state.pt")["next_step"]


def _read_step(out_dir: Path, index: int, steps: int) -> StepRecord:
    shard, local = divmod(index, steps)
    value = _load(shard_path(out_dir / "shards", shard) / "steps" / f"{name(local)}.pt")
    return _decode_record(value)


def load_step(out_dir: Path, index: int) -> StepRecord:
    """Interface: load the globally indexed record without materialising a dataset."""
    return _read_step(out_dir, index, _steps_per_shard(out_dir))


def read_history(out_dir: Path, index: int, states: int) -> RawHistory:
    """Interface: raw states restart..s_t and their preceding actions, oldest first."""
    history_start = max(0, index - states + 1)
    restart = (history_start // states) * states
    steps = _steps_per_shard(out_dir)
    observations, actions = [], []
    for current in range(restart, index + 1):
        if current == 0:
            observations.append(_read_step(out_dir, 0, steps).before)
            actions.append(None)
        else:
            previous = _read_step(out_dir, current - 1, steps)
            observations.append(previous.after)
            actions.append(previous.action)
    return RawHistory(observations, actions, restart)
