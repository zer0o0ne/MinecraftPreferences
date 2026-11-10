"""Quantised MCTS outcomes, full-root ranks and support-sensitive divergences."""

from types import SimpleNamespace

import numpy as np
import torch

from diagnostics.core.world import config, context
from mcpref.diagnostics.producers.mcts import on_step_record


def test_step_record_quantised_hits_full_probabilities_and_mask():
    root = {
        "z": torch.zeros(2),
        "children": [
            {"action": 0, "prior": 0.6, "x": torch.tensor([0.04, 0.0])},
            {"action": 1, "prior": 0.4, "x": torch.tensor([2.0, 0.0])},
        ],
    }
    record = SimpleNamespace(
        index=7,
        tree={"root": root},
        probabilities=torch.tensor([0.75, 0.25]),
        policy_probabilities=torch.tensor([0.6, 0.4]),
        action=0,
        delta=torch.tensor([0.04, 0.0]),
        predicted=torch.tensor([0.04, 0.0]),
        before=SimpleNamespace(mask=np.array([True, False])),
    )
    ctx = context(
        "sampling/shards/000000/steps/000007.pt", "mcts", steps=(7,), clusters=("e0",)
    )
    ctx = ProducerContext(
        **{**ctx.__dict__, "filters": {"alpha": [1.0, 1.0], "epsilon": 0.1, "width": 1}}
    )
    result = on_step_record(record, cfg=config(), context=ctx)
    metrics = result["metrics"]
    assert metrics["mcts.chosen_hit"]["value"] == 1.0
    assert metrics["mcts.topk_hit"]["value"] == 1.0
    assert metrics["mcts.support_hit"]["value"] == 1.0
    assert metrics["mcts.matched_mass"]["mean"] == 0.75
    assert metrics["mcts.top1_prior_agreement"]["value"] == 1.0
    assert metrics["mcts.top1_policy_agreement"]["value"] == 1.0
    assert metrics["mcts.rank_prior"]["mean"] == 1.0
    assert abs(metrics["mcts.policy_support_mass"]["mean"] - 1.0) < 1e-6
    assert abs(metrics["mask.probability"]["mean"] - 0.4) < 1e-6
    assert metrics["mask.executed_violation"]["value"] == 0.0


from mcpref.diagnostics.core.types import ProducerContext
