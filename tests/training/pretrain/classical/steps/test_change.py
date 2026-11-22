"""Independent occurrence-wise oracle for classical change-loss routing and gradients."""

import copy
from dataclasses import replace

import pytest
import torch
from models.policy.policy_oracle import cfg_of
from models.policy.policy_world import action_masks
from torch.testing import assert_close

from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from mcpref.training.pretrain.classical.steps.loss import step_gradients
from mcpref.training.pretrain.classical.steps.micro_loss import micro_loss
from training.pretrain.classical.loop_modules import D, fakes
from training.pretrain.classical.loop_world import clip, p, u
from training.pretrain.classical.steps.test_loss import CFG, assert_grads, grads

M, N_C, N_R = 7, 11, 5
W_PHI, W_P = 0.4, 1.3
ITEMS = [u(0), p(0), u(1), u(2)]
ACTIONS = {
    "u0": [["forward", "forward", "attack"], ["forward", "use", "use"]],
    "u1": [[], ["use", "use"]],
    "u2": [["forward"], ["jump"]],
}


def read(path):
    value = clip(path)
    actions = ACTIONS.get(value.info["id"], [["attack"], ["attack"]])
    value.info.update(actions=actions, n=list(map(len, actions)), pitch=[0.0] * 3)
    return value


@pytest.fixture
def modules(tmp_path):
    masks = action_masks(tmp_path / "masks")
    cfg = cfg_of(action_dropout=0.0)
    result = fakes(noise=False, stateful=False)
    result["pi"] = Policy(cfg, D, masks.names).double()
    result["pi_loss"] = PolicyLoss(
        cfg, masks, lambda delta: torch.linalg.vector_norm(delta, dim=-1)
    )
    return result


def policy_terms(modules, clips, Z):
    """Every occurrence contributes once; only membership of the prior list matters."""
    logits, shifts, _, probability = modules["pi"](Z, clips)
    nll = -logits.log_softmax(-1)
    base, auxiliary = [], []
    for i, value in enumerate(clips):
        action, shift, change = Z.new_zeros(3).unbind()
        intervals = value.info["actions"]
        for t, names in enumerate(intervals):
            for name in names:
                loss = nll[i, t, modules["pi"].names.index(name)] / len(names)
                action = action + loss
                if t:
                    repeated = name in intervals[t - 1]
                    error = probability[i, t] if repeated else 1 - probability[i, t]
                    change = change + error * loss
            if len(names) == 1:
                target = (Z[i, t + 1] - Z[i, t]).detach()
                shift = shift + (shifts[i, t] - target).square().sum()
        base.append((action + modules["pi_loss"].shift_weight * shift) / len(intervals))
        auxiliary.append(change / len(intervals))
    return torch.stack(base), torch.stack(auxiliary)


def objective(modules, items, w_pi, w_change):
    own = [read(item["clip"]) for item in items if "clip" in item]
    Z = modules["phi"](own)
    phi = modules["phi_loss"].terms(Z, own)[0]["total"]
    pi, change = policy_terms(modules, own, Z)
    loss = M / N_C * (W_PHI * phi + w_pi * pi).sum()
    if w_change:
        loss = loss + M / N_C * w_change * change.sum()
    for item in items:
        if "clip" in item:
            continue
        pair = modules["phi"]([read(item["a"]), read(item["b"])])
        out = modules["P"](pair[:1], pair[1:])
        loss = loss + M / N_R * W_P * modules["P_loss"](out, [item["labels"]]).sum()
    return loss, pi, change


def step(modules, micro, w_pi, w_change, items=ITEMS):
    return step_gradients(
        items,
        modules,
        read,
        M,
        N_C,
        N_R,
        W_PHI,
        w_pi,
        W_P,
        micro,
        cfg=replace(CFG, w_change=w_change),
    )


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("w_pi", [0.25, 2.5])
@pytest.mark.parametrize("w_change", [0.0, 1.7])
def test_independent_change_weight_and_frozen_phi(modules, frozen, w_pi, w_change):
    if frozen:
        for name in ("phi", "phi_loss"):
            modules[name].requires_grad_(False)
    expected = copy.deepcopy(modules)
    want, base, change = objective(expected, ITEMS, w_pi, w_change)
    want.backward()
    got = step(modules, 2, w_pi, w_change)
    assert got["loss"] == pytest.approx(want.item(), rel=1e-12)
    assert got["pi"] == pytest.approx(base.detach().mean().item(), rel=1e-12)
    assert got["components"]["pi"]["change"] == pytest.approx(
        change.detach().mean().item(), rel=1e-12
    )
    assert_grads(grads(modules), grads(expected))
    change_gradient = modules["pi"].change_head.weight.grad
    assert (change_gradient is None) == (w_change == 0)
    if w_change:
        assert change_gradient.abs().sum() > 0
    assert (modules["phi"].A.grad is None) == frozen


@pytest.mark.parametrize("micro", [1, 3, 6])
def test_auxiliary_microbatch_sum_uses_standalone_population(modules, micro):
    expected = copy.deepcopy(modules)
    want, _, change = objective(expected, ITEMS, 0.8, 2.3)
    want.backward()
    got = step(modules, micro, 0.8, 2.3)
    assert got["counts"]["clips"] == 3 and got["counts"]["pairs"] == 1
    assert got["loss"] == pytest.approx(want.item(), rel=1e-12)
    assert got["components"]["pi"]["change"] == pytest.approx(
        change.detach().mean().item(), rel=1e-12
    )
    assert_grads(grads(modules), grads(expected))


@pytest.mark.parametrize("w_change", [0.0, 1.7])
def test_weighted_pi_diagnostic_contains_auxiliary_gradient(modules, w_change):
    expected = copy.deepcopy(modules)
    clips = [read(u(i)["clip"]) for i in range(3)]
    base, change = policy_terms(expected, clips, expected["phi"](clips))
    want = M / N_C * 0.3 * base.sum()
    if w_change:
        want = want + M / N_C * w_change * change.sum()
    want.backward()
    _, result = micro_loss(
        modules, clips, [], (M / N_C, W_PHI, 0.3, 0.0), w_change=w_change
    )
    assert_close(result.weighted["pi"], want)
    result.weighted["pi"].backward()
    assert_grads(grads(modules), grads(expected))


def test_pair_only_has_no_auxiliary_policy_pass(modules):
    got = step(modules, 1, 0.8, 2.3, items=[p(0)])
    assert got["pi"] is None and got["components"]["pi"] is None
    assert all(parameter.grad is None for parameter in modules["pi"].parameters())


def test_reference_reports_unweighted_change_separately_from_base_pi(modules):
    from mcpref.training.pretrain.classical.references.metrics import (
        empty_part,
        finish_part,
    )
    from mcpref.training.pretrain.classical.references.pairs import evaluate_pair
    from training.pretrain.classical.references.test_reference import fit

    cfg = replace(CFG, reference_pairs=1, w_pi=0.3, w_change=2.7)
    part = empty_part(("joint",))
    item = dict(p(0), a=u(0)["clip"], b=u(1)["clip"], part="train")
    clips = [read(item["a"]), read(item["b"])]
    with torch.no_grad():
        base, change = policy_terms(modules, clips, modules["phi"](clips))
        evaluate_pair(
            cfg, modules, item, 0, read, None, None, [], {"train": part}, None
        )
        result = finish_part(cfg, modules, fit(), part, ("joint",))["joint"]["losses"]
    assert change.mean() > 0
    assert result["pi_change"] == pytest.approx(change.mean().item(), rel=1e-12)
    assert result["pi"] == pytest.approx(base.mean().item(), rel=1e-12)
