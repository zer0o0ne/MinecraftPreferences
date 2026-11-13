"""Micro-batches reach real Qwen as independent, padded batch rows."""

import pytest
import torch

from mcpref.models.perception.phi import Phi
from mcpref.models.perception.precision import enable_fp8_mlp
from models.perception.test_phi import phi64
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)

pytestmark = runs_backbone


def test_four_clips_use_one_backbone_forward(tmp_path):
    phi = phi64(tmp_path)
    clips = [random_clip(i, [0, 1], [False, False]) for i in range(4)]
    calls = []
    handle = phi.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["input_ids"].shape[0]),
        with_kwargs=True,
    )
    try:
        assert phi(clips).shape == (4, 2, 4)
    finally:
        handle.remove()
    assert calls == [4]


@pytest.mark.parametrize("lengths", [(1, 3, 2, 1), (2, 2, 2, 2)])
def test_padding_outputs_and_gradients_match_separate_streams(tmp_path, lengths):
    phi = phi64(tmp_path, marker="##")
    clips = [
        random_clip(i, list(range(length)), [False] * length)
        for i, length in enumerate(lengths)
    ]
    states = phi.prepare(clips)
    rng = torch.get_rng_state()
    actual = phi.batch_frames_z(states)
    assert torch.equal(rng, torch.get_rng_state())
    sum((i + 1) * z.square().sum() for i, z in enumerate(actual)).backward()
    gradients = {
        n: p.grad.clone() for n, p in phi.named_parameters() if p.grad is not None
    }
    phi.zero_grad(set_to_none=True)
    reference = [phi.frames_z(*state) for state in states]
    sum((i + 1) * z.square().sum() for i, z in enumerate(reference)).backward()
    for z, expected in zip(actual, reference):
        torch.testing.assert_close(z, expected)
    for name, parameter in phi.named_parameters():
        if name in gradients:
            # Qwen's delta-rule and RMSNorm have internal FP32 reductions.
            torch.testing.assert_close(
                gradients[name], parameter.grad, rtol=2e-5, atol=2e-7
            )
        else:
            assert parameter.grad is None


def test_other_batch_rows_do_not_change_a_clip_or_its_gradient(tmp_path):
    phi = phi64(tmp_path)
    clips = [random_clip(i, [0, 1], [False, False]) for i in range(4)]
    output = phi(clips)[2]
    output.sum().backward()
    grads = {n: p.grad.clone() for n, p in phi.named_parameters() if p.grad is not None}
    phi.zero_grad(set_to_none=True)
    alone = phi.clip_z(clips[2])
    alone.sum().backward()
    torch.testing.assert_close(output, alone)
    for name, parameter in phi.named_parameters():
        if name in grads:
            torch.testing.assert_close(
                grads[name], parameter.grad, rtol=2e-5, atol=2e-7
            )


@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp8-aot"])
def test_bfloat16_and_fp8_padded_batch_checkpoint_backward(tmp_path, precision):
    phi = Phi(backbone(torch.bfloat16), tokenizer(), config(tmp_path)).bfloat16()
    if precision != "bf16":
        enable_fp8_mlp(phi.backbone)
    if precision == "fp8-aot":
        for layer in phi.backbone.language_model.layers:
            layer.mlp.compile(backend="aot_eager", fullgraph=True, dynamic=True)
    clips = [random_clip(i, [0, 1], [False, False]) for i in range(4)]
    actual = phi(clips)
    expected = torch.stack([phi.clip_z(clip) for clip in clips])
    torch.testing.assert_close(actual, expected)
    actual.float().square().sum().backward()
    for group in (phi.backbone.visual, phi.backbone.language_model, phi.head):
        grads = [p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert all(g.dtype == torch.bfloat16 for g in grads)
