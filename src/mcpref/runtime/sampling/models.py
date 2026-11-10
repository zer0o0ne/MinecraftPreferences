"""Construction and stream persistence of specs/env-sampling.md (1), Datasets."""

import copy
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from mcpref.config import REPO
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.infrastructure.runs.files import finished_run, sha256
from mcpref.models.factory import build_modules
from mcpref.models.perception.config import load_config as load_perception_config
from mcpref.models.perception.phi import Phi
from mcpref.models.perception.stream import PhiStream
from mcpref.models.policy.config import load_config as load_policy_config
from mcpref.models.policy.network import Policy
from mcpref.models.preference.config import load_config as load_preference_config
from mcpref.models.preference.network import Preference
from mcpref.runtime.sampling.codecs import _cpu
from mcpref.runtime.sampling.config import SamplingConfig
from mcpref.training.pretrain.config import load_config as load_pretrain_config


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
        perception,
        masks,
        policy,
        preference,
        paired=True,
        device=pretrain.device,
        latent_init_scale=pretrain.latent_init_scale,
    )
    path = REPO / pretrain.out_dir / "model.pt"
    weights = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("phi", "pi", "P"):
        built[name].load_state_dict(weights[name])
        built[name].eval()
    return SamplingModules(built["phi"], built["pi"], built["P"], sha256(path))


_CACHE_FIELDS = (
    "keys",
    "values",
    "sliding",
    "cumulative_length",
    "conv_states",
    "recurrent_states",
)


def _linear_state(layer: object, name: str) -> torch.Tensor | None:
    states = getattr(layer, name, None)
    return None if states is None else states[0]


def encode_stream(stream: PhiStream) -> dict:
    """Datasets: primitive ids/first line, CPU grids and complete per-layer caches."""
    cache = []
    for layer in stream.cache.layers:
        cumulative_length = getattr(layer, "cumulative_length", None)
        cache.append(
            {
                "keys": _cpu(getattr(layer, "keys", None)),
                "values": _cpu(getattr(layer, "values", None)),
                "sliding": _cpu(getattr(layer, "_sliding_window_tensor", None)),
                "cumulative_length": (
                    None if cumulative_length is None else int(cumulative_length)
                ),
                "conv_states": _cpu(_linear_state(layer, "conv_states")),
                "recurrent_states": _cpu(_linear_state(layer, "recurrent_states")),
            }
        )
    return {
        "ids": list(stream.ids),
        "grids": [_cpu(grid) for grid in stream.grids],
        "cache": cache,
        "first": copy.deepcopy(stream.first),
    }


def _cache_entries(value: dict) -> list[dict]:
    try:
        return [
            {name: layer[name] for name in _CACHE_FIELDS} for layer in value["cache"]
        ]
    except KeyError:
        raise RuntimeError("incompatible sampling state") from None


def _decode_cache(phi: Phi, entries: list[dict], device: torch.device) -> DynamicCache:
    layers = []
    for layer in entries:
        item = tuple(
            None if layer[name] is None else layer[name].to(device=device)
            for name in ("keys", "values", "sliding")
        )
        layers.append(item if item[2] is not None else item[:2])
    cache = DynamicCache(ddp_cache_data=layers, config=phi.backbone.config)
    for layer_idx, layer in enumerate(entries):
        conv, recurrent = layer["conv_states"], layer["recurrent_states"]
        if conv is not None:
            cache.update_conv_state(
                conv.to(device=device),
                layer_idx=layer_idx,
                state_idx=0,
                conv_kernel_size=conv.shape[-1],
            )
        if recurrent is not None:
            cache.update_recurrent_state(
                recurrent.to(device=device), layer_idx=layer_idx, state_idx=0
            )
        if layer["cumulative_length"] is not None:
            cache.layers[layer_idx].cumulative_length = layer["cumulative_length"]
    return cache


def decode_stream(
    phi: Phi, states: int, value: dict, device: torch.device
) -> PhiStream:
    """Datasets: restore a stream, cache tensors on device and grids on the CPU."""
    stream = PhiStream(phi, states)
    stream.ids = list(value["ids"])
    stream.grids = [grid.to(device="cpu") for grid in value["grids"]]
    stream.cache = _decode_cache(phi, _cache_entries(value), device)
    stream.first = copy.deepcopy(value["first"])
    return stream
