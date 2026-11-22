"""Explicit inference/warm-start loading of policy weights predating change_head."""

import copy
from collections.abc import Mapping

import torch
from torch import Tensor


def load_policy_weights(policy: torch.nn.Module, weights: Mapping[str, Tensor]) -> None:
    """Keep only a missing change head's seeded initialization; otherwise load strictly."""
    current = policy.state_dict()
    key = "change_head.weight"
    if key in current and key not in weights:
        weights = copy.copy(weights)
        weights[key] = current[key]
    policy.load_state_dict(weights, strict=True)
