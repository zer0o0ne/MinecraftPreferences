"""Update-health hand oracles, including unsupported and zero-gradient rows."""

import math

import torch

from mcpref.diagnostics.observations import UpdateObservations
from mcpref.diagnostics.update import on_update_observation

from .world import config, context


def test_update_group_and_head_row_norms_and_unavailable_update():
    gradients = {
        "pi.head.weight": torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
        "phi": torch.tensor([2.0]),
    }
    observation = UpdateObservations(
        gradients=gradients,
        updates={"pi.head.weight": None, "phi": torch.tensor([0.5])},
        groups={"pi": ("pi.head.weight",), "phi": ("phi",)},
        clip_coefficient=0.25,
        learning_rate=0.01,
        schedule_position=4,
        logits=torch.tensor([[3.0, 4.0]]),
        supported_actions=torch.tensor([True, False]),
    )
    record = on_update_observation(
        observation,
        cfg=config(),
        context=context(
            "rl/checkpoints/000000/losses.json",
            "update",
            steps=(4,),
            clusters=("a0", "a1"),
        ),
    )
    metrics = record["metrics"]
    assert metrics["update.group.pi.preclip_norm"]["mean"] == 5.0
    assert math.isclose(metrics["update.preclip_norm"]["mean"], math.sqrt(29))
    assert metrics["update.group.pi.norm"]["mean"] is None
    assert metrics["update.group.pi.norm"]["reason"] == "zero_decay_factor"
    assert metrics["update.head_row_norm"]["mean"] == 2.5
    assert metrics["update.zero_gradient_row"]["value"] == 0.5
    assert metrics["update.unsupported_action"]["value"] == 0.5
    assert metrics["update.zero_unsupported_row"]["value"] == 0.5
    assert metrics["update.logit_norm"]["mean"] == 5.0
