"""Own clip losses and optimizer state are independent of pair multiplicity."""

import copy

import pytest
import torch

from mcpref.training.pretrain.classical.steps.loss import step_gradients
from training.pretrain.classical.loop_modules import fakes
from training.pretrain.classical.loop_world import p, u
from training.pretrain.classical.steps.test_loss import CFG, call, grads, reader


def test_pair_only_does_not_compute_own_losses():
    modules = fakes(noise=False, stateful=False)
    latents = []

    def retain(module, args, output):
        output.retain_grad()
        latents.append(output)

    modules["phi"].register_forward_hook(retain)
    result = call([p(0)], modules, 1)
    assert all(parameter.grad is None for parameter in modules["pi"].parameters())
    assert all(parameter.grad is None for parameter in modules["phi_loss"].parameters())
    assert modules["pi"].calls == modules["phi_loss"].calls == 0
    assert modules["phi"].A.grad.abs().sum() > 0
    assert modules["P"].u.grad.abs().sum() > 0
    assert all(row.abs().sum() > 0 for row in latents[0].grad)
    assert result["phi"] is result["pi"] is None
    assert result["components"]["phi"] is result["components"]["pi"] is None
    assert result["counts"]["clips"] == result["counts"]["action_targets"] == 0


@pytest.mark.parametrize("micro", [1, 2, 3, 8])
def test_more_pairs_preserve_own_epoch_objective_and_gradients(micro):
    pair = p(0)
    own = [{"clip": pair[side]} for side in "ab"]
    reverse = {"a": pair["b"], "b": pair["a"], "labels": {"p": 0.0, "inc": 0}}
    results = []
    for pairs, steps in (([], 1), ([pair], 2), ([pair, reverse, pair], 4)):
        modules = fakes(noise=False, stateful=False)
        result = step_gradients(
            own + pairs,
            modules,
            reader(),
            steps,
            len(own),
            len(pairs),
            0.4,
            0.7,
            0.0,
            micro,
            cfg=CFG,
        )
        own_gradients = {
            name: value / steps
            for name, value in grads(modules).items()
            if not name.startswith("P") and value is not None
        }
        results.append((result["loss"] / steps, own_gradients, result))
    expected, expected_gradients, logged = results[0]
    for actual, actual_gradients, result in results[1:]:
        assert actual == pytest.approx(expected, rel=1e-12)
        assert result["phi"] == pytest.approx(logged["phi"], rel=1e-12)
        assert result["pi"] == pytest.approx(logged["pi"], rel=1e-12)
        assert result["counts"]["clips"] == len(own)
        assert actual_gradients.keys() == expected_gradients.keys()
        for name in expected_gradients:
            torch.testing.assert_close(actual_gradients[name], expected_gradients[name])


def test_pair_only_preserves_own_parameters_buffers_and_initialized_adam_state():
    modules = fakes(noise=False, stateful=True)
    parameters = [p for module in modules.values() for p in module.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=0.1,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.2,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    call([u(0)], modules, 1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    names = ("pi", "pi_loss", "phi_loss")
    before = {name: copy.deepcopy(modules[name].state_dict()) for name in names}
    states = {
        parameter: copy.deepcopy(optimizer.state[parameter])
        for name in names
        for parameter in modules[name].parameters()
    }
    assert states and all(state["step"] == 1 for state in states.values())
    call([p(0), p(1)], modules, 1)
    optimizer.step()
    for name in names:
        for key, value in modules[name].state_dict().items():
            assert torch.equal(value, before[name][key])
    for parameter, state in states.items():
        assert parameter.grad is None
        for key, value in state.items():
            assert torch.equal(optimizer.state[parameter][key], value)
