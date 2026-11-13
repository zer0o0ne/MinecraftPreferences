"""Micro-batched perception with bounded, separate reward-only forwards."""

import torch

from mcpref.runtime.sampling.config import SamplingConfig
from mcpref.training.rl.learning.history import encode_refs
from mcpref.training.rl.pairs.types import StateRef
from mcpref.training.rl.ppo.config import PpoConfig


def state_ref(sampling: SamplingConfig, index: int) -> StateRef:
    """The owning trajectory's episode identity of one global action/state index."""
    cycle = index // sampling.steps
    return StateRef(f"{sampling.out_dir}#{cycle:06d}", index)


def ppo_eligible(item: dict, sampling: SamplingConfig, ppo: PpoConfig) -> bool:
    kind, t = item["kind"], item["index"]
    return (
        kind == "pending"
        or t + ppo.horizon <= (t // sampling.steps + 1) * sampling.steps
    )


def micro_histories(items, phi, perception, sampling, states, action_count, ppo, fixed):
    """Return ordered differentiable histories and no-grad reward histories."""
    refs, reward_refs = [], []
    for item in items:
        kind, t = item["kind"], item.get("index")
        if kind == "preference":
            refs.extend(StateRef(**item[key]) for key in ("a", "b"))
            continue
        refs.append(state_ref(sampling, t))
        if kind == "fresh":
            refs.append(state_ref(sampling, t + 1))
        if not fixed and ppo_eligible(item, sampling, ppo):
            reward_refs.extend(
                state_ref(sampling, t + k)
                for k in range(2 if kind == "fresh" else 1, ppo.horizon + 1)
            )
    encoded = encode_refs(refs, phi, perception, sampling, states, action_count)
    rewards = []
    with torch.no_grad():
        for start in range(0, len(reward_refs), len(items)):
            rewards.extend(
                encode_refs(
                    reward_refs[start : start + len(items)],
                    phi,
                    perception,
                    sampling,
                    states,
                    action_count,
                )
            )
    return iter(encoded), iter(rewards)
