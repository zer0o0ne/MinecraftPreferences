"""A cumulative recurrent oracle behind the production continuous stream adapter."""

import torch
import yaml

from mcpref.config import read_yaml
from mcpref.models.perception.continuous import ContinuousState
from mcpref.models.perception.continuous.state import GDNState
from runtime.sampling.helpers import modules as classical_modules


def modules(*args, **kwargs):
    result = classical_modules(*args, **kwargs)
    result.phi.backbone.config.layer_types = ["linear_attention", "full_attention"]
    return result


def setup(site, monkeypatch):
    site.modules.phi.backbone.config.layer_types = [
        "linear_attention",
        "full_attention",
    ]
    root = site.root
    path = root / site.cfg.pretrain
    original = read_yaml(path)
    classical = "configs/training/pretrain/classical/config.yaml"
    continuous = "configs/training/pretrain/continuous/config.yaml"
    for name, raw in (
        (classical, original),
        (
            continuous,
            {
                key: value
                for key, value in original.items()
                if key not in ("batch", "micro_batch")
            }
            | {
                "tbptt_states": 2,
                "preference_micro_batch": 2,
                "archive_shard_size": 4,
                "archive_cache_size": 4,
            },
        ),
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(raw))
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "continuous",
                "classical": {"config": classical, "index": "unused-classical.yaml"},
                "continuous": {"config": continuous, "index": "unused-continuous.yaml"},
            }
        )
    )
    for name, history in (("perception", 3), ("policy", 2), ("preference", 5)):
        model = root / original[name]
        model.write_text(yaml.safe_dump(read_yaml(model) | {"history": history}))
    model = root / "data/pretrain-scaled-init/model.pt"
    weights = torch.load(model, weights_only=True)
    torch.save(weights | {"pretrain_mode": "continuous"}, model)
    monkeypatch.setattr(
        "mcpref.models.perception.continuous.observation.observation_step", observe
    )


def observe(phi, frame, line, state=None):
    index = line["id"]
    phi.events.append(("stream", index))
    phi.grad_modes.append(torch.is_grad_enabled())
    old = 0 if state is None else state.layers[0].recurrent
    recurrent = phi.weight.new_tensor(index + 1) + old
    conv = phi.weight.new_tensor([index], dtype=torch.float32)
    origin = dict(line) if state is None else state.origin
    return phi.latent(index) + recurrent, ContinuousState(
        (GDNState(conv, recurrent),), origin
    )
