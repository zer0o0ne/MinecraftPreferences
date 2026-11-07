"""Equations (2)-(3): role losses, global normalization, calls, and gradient flow."""

from types import SimpleNamespace

import pytest
import torch

from mcpref.rl.loss import step_gradients

from .loss_fakes import (
    ALPHA,
    assert_gradient_close,
    expected,
    gradients,
    install,
    modules,
    ppo_config,
)

WEIGHTS = (2.0, 3.0, 0.7, 1.4)


def fresh(index):
    return {"kind": "fresh", "index": index}


def pending(index):
    return {"kind": "pending", "index": index}


def preference(a=10, b=11):
    return {"kind": "preference", "a": {"episode": "data/sampling#000000", "index": a},
            "b": {"episode": "data/sampling#000001", "index": b},
            "labels": {"p": 0.75, "inc": 0}}  # fmt: skip


def call(items, mods, micro_batch=2, m=4, n_a=2, n_delta=2, r_c=1, horizon=2):
    w_ppo, w_delta, w_metra, w_pref = WEIGHTS
    return step_gradients(
        items,
        mods,
        sampling=SimpleNamespace(steps=max(2, horizon), out_dir="data/sampling"),
        perception=SimpleNamespace(alpha=tuple(ALPHA.tolist())),
        states=3,
        alpha=ALPHA,
        ppo=ppo_config(horizon=horizon),
        discount=0.8,
        min_p=0.1,
        m=m,
        n_a=n_a,
        n_delta=n_delta,
        r_c=r_c,
        w_ppo=w_ppo,
        w_delta=w_delta,
        w_metra=w_metra,
        w_P=w_pref,
        micro_batch=micro_batch,
    )


def assert_result(result, total, means):
    assert set(result) == {"ppo", "shift", "metra", "P", "loss"}
    assert result["loss"] == pytest.approx(total.detach().item())
    for role, value in means.items():
        if value is None:
            assert result[role] is None
        else:
            assert result[role] == pytest.approx(value)


def test_all_roles_match_float64_oracle_and_parameter_gradients(tmp_path, monkeypatch):
    events = []
    install(monkeypatch, events)
    items = [fresh(0), fresh(1), pending(2), preference()]
    actual = modules(events)
    result = call(items, actual)
    actual_grads = gradients(actual)
    reference = modules()
    total, means = expected(items, reference, 4, 2, 2, 1, WEIGHTS)
    total.backward()
    assert_result(result, total, means)
    assert_gradient_close(actual_grads, gradients(reference))
    assert actual["pi"].shift_calls == 2
    assert len([event for event in events if event[0] == "readout"]) == 3
    assert len([event for event in events if event[0] == "P"]) == 5
    reward_calls = [event for event in events if event[0] == "P"][:4]
    assert all(event[1] > event[2] for event in reward_calls)
    assert all(role in actual_grads for role in ["phi.scale", "metra.log_lambda"])
    assert any(name.startswith("pi.") for name in actual_grads)
    assert any(name.startswith("P.") for name in actual_grads)


def test_microbatch_partition_does_not_change_loss_or_gradients(monkeypatch):
    items = [fresh(0), fresh(1), pending(2), preference()]
    results = []
    for micro in (1, 3):
        install(monkeypatch, [])
        mods = modules()
        result = call(items, mods, micro_batch=micro)
        results.append((result, gradients(mods)))
    assert results[0][0] == pytest.approx(results[1][0])
    assert_gradient_close(results[0][1], results[1][1])


def test_zero_ppo_and_preference_denominators_are_absent_not_nan(monkeypatch):
    install(monkeypatch, [])
    mods = modules()
    result = call([fresh(1)], mods, n_a=0, n_delta=1, r_c=0)
    reference = modules()
    total, means = expected([fresh(1)], reference, 4, 0, 1, 0, WEIGHTS)
    total.backward()
    assert_result(result, total, means)
    assert result["ppo"] is None and result["P"] is None
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert_gradient_close(gradients(mods), gradients(reference))
    assert all(parameter.grad is None for parameter in mods["P"].parameters())


def test_pending_computes_only_ppo_and_never_retrains_shift_or_metra(monkeypatch):
    install(monkeypatch, [])
    mods = modules()
    result = call([pending(2)], mods, n_a=1, n_delta=0, r_c=0)
    reference = modules()
    total, means = expected([pending(2)], reference, 4, 1, 0, 0, WEIGHTS)
    total.backward()
    assert_result(result, total, means)
    assert_gradient_close(gradients(mods), gradients(reference))
    assert mods["pi"].shift_calls == 0
    assert mods["metra"].log_lambda.grad is None
    assert all(parameter.grad is None for parameter in mods["P"].parameters())


def test_preference_only_reaches_preference_and_phi_not_policy_or_dual(monkeypatch):
    install(monkeypatch, [])
    mods = modules()
    result = call([preference()], mods, n_a=0, n_delta=0, r_c=1)
    reference = modules()
    total, means = expected([preference()], reference, 4, 0, 0, 1, WEIGHTS)
    total.backward()
    assert_result(result, total, means)
    assert_gradient_close(gradients(mods), gradients(reference))
    assert all(parameter.grad is None for parameter in mods["pi"].parameters())
    assert mods["metra"].log_lambda.grad is None
    assert mods["phi"].scale.grad is not None and mods["P"].weight.grad is not None


def test_existing_gradients_accumulate_without_zeroing(monkeypatch):
    install(monkeypatch, [])
    mods = modules()
    for parameter in mods["phi"].parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    call([preference()], mods, n_a=0, n_delta=0, r_c=1)
    reference = modules()
    total, _ = expected([preference()], reference, 4, 0, 0, 1, WEIGHTS)
    total.backward()
    torch.testing.assert_close(
        mods["phi"].scale.grad, reference["phi"].scale.grad + 0.25
    )


def test_epoch_mean_is_global_role_normalization_with_unequal_role_counts(monkeypatch):
    batches = [[fresh(0), preference()], [fresh(1), pending(2)]]
    losses = []
    for batch in batches:
        install(monkeypatch, [])
        losses.append(call(batch, modules(), m=2)["loss"])
    reference = modules()
    global_sum, _ = expected([item for batch in batches for item in batch],
                             reference, 1, 2, 2, 1, WEIGHTS)  # fmt: skip
    assert sum(losses) / 2 == pytest.approx(global_sum.detach().item())
