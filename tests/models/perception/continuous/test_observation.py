"""Real tiny multimodal Phi observation, origin, inference and update transitions."""

import io

import torch
from models.perception import oracle
from models.perception.world import (
    backbone,
    config,
    random_clip,
    runs_backbone,
    tokenizer,
)

from mcpref.models.perception.continuous import (
    ContinuousPhiStream,
    detach_state,
    observation_step,
)
from mcpref.models.perception.meta import meta_text
from mcpref.models.perception.phi import Phi

pytestmark = runs_backbone


def world(tmp_path):
    phi = Phi(backbone(), tokenizer(), config(tmp_path)).double()
    clip = random_clip(13, [2, 3, 5], [True, False, True])
    return phi, oracle.frames(clip), [oracle.state_line(clip, i) for i in range(3)]


def test_first_observation_matches_classical_and_origin_stays_fixed(
    tmp_path, monkeypatch
):
    from mcpref.models.perception.continuous import observation

    phi, frames, lines = world(tmp_path)
    phi.eval()
    seen = []
    original = observation.segment

    def segment(tokenizer, specials, grid, meta, marker):
        seen.append(meta)
        return original(tokenizer, specials, grid, meta, marker)

    monkeypatch.setattr(observation, "segment", segment)
    with torch.no_grad():
        expected = phi.frames_z([frames[0]], [meta_text(phi.cfg, lines[0], lines[0])])[
            0
        ]
        actual, state = observation_step(phi, frames[0], lines[0])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _, state = observation_step(phi, frames[1], lines[1], state)
    assert state.origin == lines[0]
    assert seen == [meta_text(phi.cfg, line, lines[0]) for line in lines[:2]]
    assert len(state.layers) == 1


def test_inference_reset_matches_functional_path_and_state_roundtrip(tmp_path):
    phi, frames, lines = world(tmp_path)
    phi.eval()
    stream = ContinuousPhiStream(phi)
    memory = None
    with torch.no_grad():
        for frame, line in zip(frames, lines):
            expected, memory = observation_step(phi, frame, line, memory)
            torch.testing.assert_close(
                stream.step(frame, line), expected, rtol=0, atol=0
            )
            assert not expected.requires_grad
        saved = io.BytesIO()
        torch.save(detach_state(memory, device="cpu"), saved)
        saved.seek(0)
        restored = torch.load(saved, weights_only=False)
        a, _ = observation_step(phi, frames[0], lines[0], restored)
        b, _ = observation_step(phi, frames[0], lines[0], memory)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        stream.restart()
        expected, _ = observation_step(phi, frames[-1], lines[-1])
        torch.testing.assert_close(
            stream.step(frames[-1], lines[-1]), expected, rtol=0, atol=0
        )


def test_k1_checkpointed_backward_update_and_detached_state(tmp_path):
    phi, frames, lines = world(tmp_path)
    phi.train()
    optimizer = torch.optim.AdamW(phi.parameters(), lr=0.001)
    state = None
    names = tuple(phi.state_dict())
    for frame, line in zip(frames, lines):
        optimizer.zero_grad(set_to_none=True)
        z, state = observation_step(phi, frame, line, state)
        z.square().sum().backward()
        assert phi.backbone.visual.blocks[0].attn.qkv.weight.grad is not None
        assert (
            phi.backbone.language_model.layers[0].linear_attn.in_proj_qkv.weight.grad
            is not None
        )
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in phi.parameters()
        )
        optimizer.step()
        state = detach_state(state)
        assert all(
            not s.conv.requires_grad and not s.recurrent.requires_grad
            for s in state.layers
        )
    assert tuple(phi.state_dict()) == names
