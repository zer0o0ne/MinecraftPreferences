"""Real Qwen RL batches preserve restart boundaries, losses and no-grad rewards."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from models.perception.test_phi import phi64
from models.perception.world import line, runs_backbone
from runtime.sampling.helpers import state

from mcpref.runtime.sampling.shards import RawHistory
from mcpref.training.rl.learning.history import encode_ref, encode_refs
from mcpref.training.rl.pairs.types import StateRef
from training.rl.learning.loss_fakes import gradients, load_step, modules
from training.rl.learning.test_loss import call, fresh, pending, preference

pytestmark = runs_backbone


def raw(index, states):
    start = max(0, index - states + 1) // states * states
    observations = [
        replace(
            state(i),
            frame=np.full((32, 64, 3), 15 * i, dtype=np.uint8),
            line=line(
                xpos=float(i), inventory=[{"type": "dirt" * (i + 1), "quantity": i}]
            ),
        )
        for i in range(start, index + 1)
    ]
    actions = [None if i == 0 else (i - 1) % 3 for i in range(start, index + 1)]
    return RawHistory(observations, actions, start)


def test_reference_batch_matches_independent_restarts_and_action_tails(
    tmp_path, monkeypatch
):
    phi = phi64(tmp_path, alpha=[1.5, 0.5])
    monkeypatch.setattr(
        "mcpref.training.rl.learning.history.read_history",
        lambda path, index, states: raw(index, states),
    )
    sampling = SimpleNamespace(out_dir="data/sampling")
    refs = [StateRef("data/sampling#000000", i) for i in (4, 0, 5, 3)]
    actual = encode_refs(refs, phi, phi.cfg, sampling, 3, 3)
    for ref, value in zip(refs, actual):
        expected = encode_ref(ref, phi, phi.cfg, sampling, 3, 3)
        torch.testing.assert_close(value.Y, expected.Y)
        torch.testing.assert_close(value.C, expected.C, rtol=0, atol=0)


@pytest.mark.parametrize("micro", [1, 4])
def test_mixed_rl_microbatch_matches_scalar_encoding(tmp_path, monkeypatch, micro):
    mods = modules()
    mods["phi"] = phi64(tmp_path, alpha=[1.5, 0.5])
    reference = copy.deepcopy(mods)
    monkeypatch.setattr(
        "mcpref.training.rl.learning.history.read_history",
        lambda path, index, states: raw(index, states),
    )
    monkeypatch.setattr("mcpref.training.rl.learning.loss.load_step", load_step)
    # The loss test wrapper supplies alpha only; raw encoding also needs meta config.
    real_encode = encode_refs
    monkeypatch.setattr(
        "mcpref.training.rl.learning.encoding.encode_refs",
        lambda refs, phi, perception, *args: real_encode(refs, phi, phi.cfg, *args),
    )
    items = [fresh(0), fresh(1), pending(2), preference(4, 5), fresh(1)]
    calls = []
    handle = mods["phi"].backbone.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(
            (torch.is_grad_enabled(), kwargs["input_ids"].shape[0])
        ),
        with_kwargs=True,
    )
    try:
        result = call(items, mods, micro_batch=micro, n_delta=3, horizon=4)
    finally:
        handle.remove()
    assert any(batch > 1 for enabled, batch in calls if enabled)
    assert all(batch <= micro for enabled, batch in calls if not enabled)
    assert any(not enabled for enabled, batch in calls)
    assert mods["phi"].training
    # Independent old scalar encoder exercises the same complete loss/backward path.
    monkeypatch.setattr(
        "mcpref.training.rl.learning.encoding.encode_refs",
        lambda refs, phi, perception, *args: [
            encode_ref(r, phi, phi.cfg, *args) for r in refs
        ],
    )
    expected = call(items, reference, micro_batch=micro, n_delta=3, horizon=4)
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-7)
    actual_grads, expected_grads = gradients(mods), gradients(reference)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        torch.testing.assert_close(
            actual_grads[name], expected_grads[name], rtol=3e-5, atol=2e-6
        )
