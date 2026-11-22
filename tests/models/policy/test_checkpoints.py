"""Legacy policy weights may omit only the new change head, never other state."""

import pytest
import torch

from mcpref.models.policy.checkpoints import load_policy_weights
from mcpref.models.policy.network import Policy
from models.policy.policy_oracle import cfg_of


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_real_policy_weights_keep_only_missing_change_initialization(legacy, dtype):
    source = Policy(cfg_of(seed=11, action_dropout=0), 2, ("a", "b")).to(dtype).eval()
    target = Policy(cfg_of(seed=19, action_dropout=0), 2, ("a", "b")).to(dtype).eval()
    weights = source.state_dict()
    expected_head = target.change_head.weight.detach().clone()
    if legacy:
        del weights["change_head.weight"]
    else:
        expected_head = source.change_head.weight
    keys = tuple(weights)
    rng = torch.get_rng_state()
    load_policy_weights(target, weights)
    assert tuple(weights) == keys
    assert torch.get_rng_state().equal(rng)
    assert target.change_head.weight.equal(expected_head)
    for name, expected in weights.items():
        assert target.state_dict()[name].equal(expected)
    Y = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]], dtype=dtype)
    C = torch.tensor([[[0, 0, 1], [1, 0, 0]]], dtype=dtype)
    actions = torch.tensor([[0, 1]], dtype=torch.int64)
    original, loaded = source.readout(Y, C), target.readout(Y, C)
    assert source.logits(original).equal(target.logits(loaded))
    assert source.shift(original, actions).equal(target.shift(loaded, actions))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "change_shape"])
def test_compatibility_never_hides_other_key_or_shape_errors(legacy, fault):
    policy = Policy(cfg_of(), 2, ("a", "b"))
    weights = policy.state_dict()
    if legacy:
        del weights["change_head.weight"]
    if fault == "missing":
        del weights["head.weight"]
    elif fault == "unexpected":
        weights["change_head.bias"] = torch.zeros(1)
    elif fault == "shape":
        weights["embed.weight"] = torch.zeros(2, 2)
    else:
        weights["change_head.weight"] = torch.zeros(2, 8)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        load_policy_weights(policy, weights)
