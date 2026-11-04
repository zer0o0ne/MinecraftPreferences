"""Construction and stream persistence of specs/env-sampling.md (1), Datasets."""

import copy
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from mcpref import REPO
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.perception.config import load_config as load_perception_config
from mcpref.perception.phi import Phi
from mcpref.perception.stream import PhiStream
from mcpref.policy.config import load_config as load_policy_config
from mcpref.policy.network import Policy
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.preferences.network import Preference
from mcpref.pretrain.config import load_config as load_pretrain_config
from mcpref.pretrain.run import build_modules
from mcpref.runs.files import finished_run, sha256
from mcpref.sampling.codecs import _cpu
from mcpref.sampling.config import SamplingConfig


@dataclass(frozen=True)
class SamplingModules:
    """Interface: the three current networks and their common model fingerprint."""

    phi: Phi
    pi: Policy
    P: Preference
    fingerprint: str


def load_pretrained(cfg: SamplingConfig) -> SamplingModules:
    """(1): completed pretrain modules on its device, CPU-loaded weights, eval mode."""
    pretrain = load_pretrain_config(REPO / cfg.pretrain)
    finished_run(REPO / pretrain.out_dir, pretrain.out_dir, "complete")
    perception = load_perception_config(REPO / pretrain.perception)
    masks = load_masks(load_mask_config(REPO / pretrain.action_mask))
    policy = load_policy_config(REPO / pretrain.policy)
    preference = load_preference_config(REPO / pretrain.preference)
    built = build_modules(
        perception, masks, policy, preference, paired=True, device=pretrain.device
    )
    path = REPO / pretrain.out_dir / "model.pt"
    weights = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("phi", "pi", "P"):
        built[name].load_state_dict(weights[name])
        built[name].eval()
    return SamplingModules(built["phi"], built["pi"], built["P"], sha256(path))


def encode_stream(stream: PhiStream) -> dict:
    """Datasets: primitive ids/first line, CPU grids and CPU per-layer cache tuples."""
    cache = []
    for keys, values, sliding in stream.cache:
        cache.append(
            {"keys": _cpu(keys), "values": _cpu(values), "sliding": _cpu(sliding)}
        )
    return {
        "ids": list(stream.ids),
        "grids": [_cpu(grid) for grid in stream.grids],
        "cache": cache,
        "first": copy.deepcopy(stream.first),
    }


def decode_stream(
    phi: Phi, states: int, value: dict, device: torch.device
) -> PhiStream:
    """Datasets: restore a stream, cache tensors on device and grids on the CPU."""
    stream = PhiStream(phi, states)
    stream.ids = list(value["ids"])
    stream.grids = [grid.to(device="cpu") for grid in value["grids"]]
    layers = []
    for layer in value["cache"]:
        item = tuple(
            None if layer[name] is None else layer[name].to(device=device)
            for name in ("keys", "values", "sliding")
        )
        layers.append(item if item[2] is not None else item[:2])
    stream.cache = DynamicCache(ddp_cache_data=layers, config=phi.backbone.config)
    stream.first = copy.deepcopy(value["first"])
    return stream
