"""Primitive persistence of the continuous world's fixed-size perception memory."""

import copy

import torch

from mcpref.models.perception.continuous import ContinuousPhiStream, ContinuousState
from mcpref.models.perception.continuous.state import GDNState
from mcpref.runtime.sampling.codecs import _cpu


def encode(stream: ContinuousPhiStream) -> dict:
    if stream.state is None:
        raise RuntimeError("Cannot save an uninitialized continuous sampling stream")
    return {
        "mode": "continuous",
        "version": 1,
        "origin": copy.deepcopy(stream.state.origin),
        "layers": [
            {"conv": _cpu(layer.conv), "recurrent": _cpu(layer.recurrent)}
            for layer in stream.state.layers
        ],
    }


def decode(phi, value: dict, device: torch.device) -> ContinuousPhiStream:
    try:
        if value["version"] != 1 or value["mode"] != "continuous":
            raise ValueError("unknown stream format")
        expected = phi.backbone.config.get_text_config().layer_types.count(
            "linear_attention"
        )
        if len(value["layers"]) != expected:
            raise ValueError("continuous memory does not match the model's GDN layers")
        layers = tuple(
            GDNState(row["conv"].to(device), row["recurrent"].to(device))
            for row in value["layers"]
        )
        stream = ContinuousPhiStream(phi)
        stream.state = ContinuousState(layers, copy.deepcopy(value["origin"]))
        return stream
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RuntimeError("incompatible sampling state") from None
