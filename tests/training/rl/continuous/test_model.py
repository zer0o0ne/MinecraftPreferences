"""Real Qwen replay gradients from serialized old-weight sampling memory."""

import copy
from types import SimpleNamespace

import pytest
import torch
from models.perception.continuous.test_observation import world
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous import ContinuousPhiStream, observation_step
from mcpref.runtime.sampling.anchors import initialize, read_anchor, write_anchor
from mcpref.runtime.sampling.models import encode_stream
from mcpref.training.rl.config import ContinuousConfig
from mcpref.training.rl.continuous import replay


@pytest.mark.parametrize(
    "device,dtype",
    [
        pytest.param("cpu", torch.float64, marks=runs_backbone),
        pytest.param(
            "cuda", torch.bfloat16, marks=[pytest.mark.gpu, pytest.mark.timeout(180)]
        ),
    ],
)
def test_real_qwen_stale_anchor_checkpointed_gradient_and_live_continuation(
    tmp_path, monkeypatch, device, dtype
):
    if device == "cuda":
        from models.perception.continuous.test_gpu import assert_fla

        assert_fla()
    phi, frames, lines = world(tmp_path)
    phi.to(device=device, dtype=dtype).eval()
    observations = [
        SimpleNamespace(frame=frames[i % 3], line=lines[i % 3]) for i in range(9)
    ]
    sampling = SimpleNamespace(out_dir="trajectory", steps=4)
    out = tmp_path / sampling.out_dir
    stream = ContinuousPhiStream(phi)
    for owner in range(2):
        directory = out / "shards" / f"{owner:06d}"
        directory.mkdir(parents=True)
        initialize(directory, 2, "old-weights")
    for index, obs in enumerate(observations):
        directory = out / "shards" / f"{max(0, index - 1) // 4:06d}"
        write_anchor(directory, index, stream, "old-weights", 2)
        stream.step(obs.frame, obs.line)
    saved = encode_stream(stream)
    print(
        "continuous state dtypes:",
        [(x.conv.dtype, x.recurrent.dtype) for x in stream.state.layers],
    )
    with torch.no_grad():
        phi.head[0].weight.add_(0.01)
        phi.backbone.language_model.layers[0].linear_attn.in_proj_qkv.weight.mul_(1.05)
    reference = copy.deepcopy(phi).train()
    reference.backbone.gradient_checkpointing_disable()
    phi.train()
    phi.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    monkeypatch.setattr(replay, "REPO", tmp_path)
    monkeypatch.setattr(
        replay,
        "read_observation",
        lambda _, i: (observations[i], None if i == 0 else (i - 1) % 3),
    )

    encoded, rewards = replay.branch(
        phi, sampling, 2, 3, ContinuousConfig(2, 2), 6, fresh=True, horizon=2
    )
    # Independently unroll from A_4: s_4 is burn-in; s_5,s_6,s_7 carry gradients.
    memory = read_anchor(out, 4, 4, reference, torch.device(device), 2)
    rows = []
    for i in range(4, 8):
        obs = observations[i]
        with torch.set_grad_enabled(i >= 5):
            z, memory = observation_step(reference, obs.frame, obs.line, memory)
        rows.append(z)
    expected = [torch.stack(rows[1:3])[None], torch.stack(rows[2:4])[None]]
    for actual, oracle in zip(encoded, expected, strict=True):
        torch.testing.assert_close(actual.Y, oracle, rtol=0, atol=0)
    assert len(rewards) == 1 and rewards[0].Y.grad_fn is None
    actual_loss = sum(value.Y.float().square().mean() for value in encoded)
    expected_loss = sum(value.float().square().mean() for value in expected)
    actual_loss.backward()
    expected_loss.backward()
    actual_grads = {n: p.grad for n, p in phi.named_parameters() if p.grad is not None}
    expected_grads = {
        n: p.grad for n, p in reference.named_parameters() if p.grad is not None
    }
    assert actual_grads.keys() == expected_grads.keys()
    assert any("visual" in n and g.abs().sum() > 0 for n, g in actual_grads.items())
    assert any(
        "linear_attn" in n and g.abs().sum() > 0 for n, g in actual_grads.items()
    )
    for key, gradient in actual_grads.items():
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_grads[key], rtol=0, atol=0)
    # Offline graphs never replace or mutate the live sampling memory.
    after = encode_stream(stream)
    for a, b in zip(saved["layers"], after["layers"], strict=True):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    phi.eval()
    from mcpref.runtime.sampling.continuous import decode

    resumed = decode(phi, saved, torch.device(device))
    expected_next = stream.step(frames[0], lines[0])
    actual_next = resumed.step(frames[0], lines[0])
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)
