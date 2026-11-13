"""The pretrain micro-batch, including pairs and a short tail, reaches real Qwen."""

import copy

import pytest
import torch
from models.perception.test_phi import phi64
from models.perception.world import runs_backbone

from mcpref.models.perception.loss import PhiLoss
from training.pretrain.loop_modules import fakes
from training.pretrain.steps.test_loss import call
from training.pretrain.steps.test_micro_loss import clips, gradients

pytestmark = runs_backbone


def test_mixed_microbatches_match_separate_clip_losses_and_gradients(tmp_path):
    mods = fakes(noise=False, stateful=False)
    mods["phi"] = phi64(tmp_path, alpha=[1.0, 2.0])
    mods["phi_loss"] = PhiLoss(mods["phi"].cfg).double()
    reference = copy.deepcopy(mods)
    reference["phi"].batch_frames_z = lambda states: [
        reference["phi"].frames_z(*state) for state in states
    ]
    data = dict(zip(("a", "b"), clips()))
    for name, clip in data.items():
        clip.info["relpath"] = name
    items = [
        {"clip": "a"},
        {"a": "a", "b": "b", "labels": {"p": 0.75, "inc": 0}},
        {"clip": "b"},
        {"clip": "a"},
        {"clip": "b"},
    ]
    batches = []
    handle = mods["phi"].backbone.register_forward_pre_hook(
        lambda module, args, kwargs: batches.append(kwargs["input_ids"].shape[0]),
        with_kwargs=True,
    )
    try:
        actual = call(items, mods, micro=4, read=data.__getitem__)
    finally:
        handle.remove()
    expected = call(items, reference, micro=4, read=data.__getitem__)
    assert batches == [5, 1]  # Four items include both sides of one preference pair.
    for key in ("loss", "phi", "pi", "P"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-7)
    actual_grads, expected_grads = gradients(mods), gradients(reference)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        torch.testing.assert_close(
            actual_grads[name], expected_grads[name], rtol=3e-5, atol=2e-6
        )
