"""Real tiny Qwen inference through the runtime's primitive stream persistence."""

from dataclasses import replace

import pytest
import torch
from models.perception.continuous.test_observation import world
from models.perception.world import runs_backbone

from mcpref.models.perception.continuous import ContinuousPhiStream
from mcpref.runtime.sampling.models import decode_stream, encode_stream


@pytest.mark.parametrize(
    "device,dtype",
    [
        pytest.param("cpu", torch.float64, marks=runs_backbone),
        pytest.param(
            "cuda", torch.bfloat16, marks=[pytest.mark.gpu, pytest.mark.timeout(180)]
        ),
    ],
)
def test_real_qwen_continues_past_history_with_exact_saved_state(
    tmp_path, device, dtype
):
    phi, frames, lines = world(tmp_path)
    phi.to(device=device, dtype=dtype).eval()
    phi.cfg = replace(phi.cfg, history=3)
    uninterrupted = ContinuousPhiStream(phi)
    resumed = ContinuousPhiStream(phi)
    shapes = None
    for i in range(12):
        frame, line = frames[i % 3], lines[i % 3]
        expected = uninterrupted.step(frame, line)
        actual = resumed.step(frame, line)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
        saved = encode_stream(resumed)
        current = [
            (row["conv"].shape, row["recurrent"].shape) for row in saved["layers"]
        ]
        assert shapes is None or shapes == current
        shapes = current
        assert saved["origin"] == lines[0]
        if i in (2, 6):
            path = tmp_path / "stream.pt"
            torch.save(saved, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            resumed = decode_stream(phi, 2, restored, torch.device(device))
    fresh = ContinuousPhiStream(phi).step(frames[-1], lines[-1])
    assert not torch.equal(actual, fresh)


@runs_backbone
def test_missing_gdn_memory_is_rejected_instead_of_resetting(tmp_path):
    phi, frames, lines = world(tmp_path)
    phi.eval()
    stream = ContinuousPhiStream(phi)
    stream.step(frames[0], lines[0])
    saved = encode_stream(stream)
    saved["layers"] = []
    with pytest.raises(RuntimeError, match="incompatible sampling state"):
        decode_stream(phi, 2, saved, torch.device("cpu"))
