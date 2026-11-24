"""Nonfinite gradients must not mutate weights or an existing Adam state."""

import copy
from types import SimpleNamespace

import pytest
import torch

from mcpref.training.optim import adamw, update


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("initialized", [False, True])
def test_nonfinite_gradient_aborts_before_any_optimizer_mutation(bad, initialized):
    cfg = SimpleNamespace(
        device="cpu",
        groups={"g": {"lr": 1e-5, "weight_decay": 0.01}},
        betas=(0.9, 0.999),
        adam_eps=1e-8,
        grad_clip=1.0,
        warmup_steps=100,
        lr_floor=0.1,
    )
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0])) for _ in range(2)]
    groups = {"g": parameters}
    optimizer = adamw(cfg, groups)
    if initialized:
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        update(cfg, optimizer, groups, 1, 1000)
    before = [parameter.detach().clone() for parameter in parameters]
    state = copy.deepcopy(optimizer.state_dict())
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[1].grad = torch.tensor([0.0, bad])
    with pytest.raises(RuntimeError, match="non-finite"):
        update(cfg, optimizer, groups, 2, 1000)
    for parameter, saved in zip(parameters, before):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
    restored = optimizer.state_dict()
    assert restored["param_groups"] == state["param_groups"]
    assert restored["state"].keys() == state["state"].keys()
    for key, values in state["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(
                restored["state"][key][name], value, rtol=0, atol=0
            )
