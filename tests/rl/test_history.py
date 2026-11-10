"""Equation (6): restart-aware differentiable histories and eval/no-grad adapters."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from perception import oracle
from perception.world import config as perception_config
from perception.world import line
from pretrain.index_run import repo_root
from sampling.helpers import state

from mcpref.mcts.tree import LatentHistory
from mcpref.rl.history import encode_raw, encode_ref, eval_encoder
from mcpref.rl_pairs.types import StateRef
from mcpref.sampling.shards import RawHistory


class RawPhi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.calls = []

    def frames_z(self, frames, metas):
        self.calls.append((list(frames), list(metas)))
        ids = torch.tensor(
            [float(frame[0, 0, 0]) for frame in frames], dtype=self.scale.dtype
        )
        return torch.stack((ids, ids + 0.5), dim=-1) * self.scale


def raw_history():
    states = []
    for index in range(5):
        value = dataclasses.replace(
            state(index), line=line(xpos=float(index), ypos=64.0, zpos=-float(index))
        )
        states.append(value)
    return RawHistory(states=states, actions=[None, 0, 1, 2, 0], restart=0)


def test_restart_segments_meta_origins_latents_and_preceding_actions(tmp_path):
    perception = perception_config(tmp_path)
    phi = RawPhi().train()
    result = encode_raw(phi, perception, raw_history(), states=3, action_count=3)
    assert [len(call[0]) for call in phi.calls] == [3, 2]
    values = dataclasses.asdict(perception)
    raw = raw_history()
    expected_metas = [
        [
            oracle.meta(raw.states[i].line, raw.states[start].line, values)
            for i in range(start, stop)
        ]
        for start, stop in [(0, 3), (3, 5)]
    ]
    assert [call[1] for call in phi.calls] == expected_metas
    ids = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    expected_y = torch.stack((ids, ids + 0.5), dim=-1) * phi.scale
    torch.testing.assert_close(result.Y, expected_y.unsqueeze(0))
    expected_c = torch.zeros(1, 3, 4, dtype=torch.float64)
    expected_c[0, 0, 1] = expected_c[0, 1, 2] = expected_c[0, 2, 0] = 1.0
    torch.testing.assert_close(result.C, expected_c)
    assert result.Y.shape == (1, 3, 2) and result.C.shape == (1, 3, 4)
    assert result.Y.dtype == result.C.dtype == torch.float64
    assert result.Y.device == result.C.device == phi.scale.device
    assert result.Y.requires_grad and phi.training and torch.is_grad_enabled()
    result.Y.sum().backward()
    assert phi.scale.grad is not None and phi.scale.grad != 0


def test_retained_tail_is_bitwise_exact_with_restart_and_no_action(tmp_path):
    raw = dataclasses.replace(raw_history(), actions=[0, None, 1, 2, None], restart=1)
    phi = RawPhi()
    result = encode_raw(phi, perception_config(tmp_path), raw, 4, 3)
    assert [len(call[0]) for call in phi.calls] == [3, 2]
    ids = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    expected_y = torch.stack((ids, ids + 0.5), dim=-1) * phi.scale
    expected_c = torch.zeros(1, 4, 4, dtype=torch.float64)
    expected_c[0, 0, 3] = expected_c[0, 1, 1] = 1.0
    expected_c[0, 2, 2] = expected_c[0, 3, 3] = 1.0
    assert torch.equal(result.Y, expected_y.unsqueeze(0))
    assert torch.equal(result.C, expected_c)


def test_encode_ref_validates_episode_and_reads_repository_raw_history(
    tmp_path, monkeypatch
):
    perception = perception_config(tmp_path)
    phi = RawPhi()
    calls = []
    raw = raw_history()

    def read(path, index, states):
        calls.append((path, index, states))
        return raw

    monkeypatch.setattr("mcpref.rl.history.read_history", read)
    root = tmp_path / "repo"
    sampling = SimpleNamespace(out_dir="data/sampling")
    with repo_root(root):
        result = encode_ref(StateRef("data/sampling#000012", 4), phi, perception,
                            sampling, 3, 3)  # fmt: skip
    assert calls == [(root / "data/sampling", 4, 3)]
    assert result.Y.shape == (1, 3, 2)


@pytest.mark.parametrize(
    "episode",
    [
        "data/sampling#00000",
        "data/sampling#abcdef",
        "other#000000",
        "data/sampling",
        "data/sampling#1x",
    ],
)
def test_encode_ref_rejects_invalid_episode(tmp_path, episode):
    with pytest.raises(ValueError):
        encode_ref(
            StateRef(episode, 0),
            RawPhi(),
            perception_config(tmp_path),
            SimpleNamespace(out_dir="data/sampling"),
            3,
            3,
        )


def test_eval_encoder_reference_order_no_grad_and_mode_restoration(
    tmp_path, monkeypatch
):
    phi = RawPhi().train()
    calls = []

    def encoded(ref, phi_arg, perception, sampling, states, action_count):
        assert phi_arg is phi and not phi.training and not torch.is_grad_enabled()
        calls.append(ref)
        y = phi.scale * torch.tensor([[[float(ref.index), 0.5]]], dtype=torch.float64)
        return LatentHistory(y, torch.zeros(1, 1, 4, dtype=torch.float64))

    monkeypatch.setattr("mcpref.rl.history.encode_ref", encoded)
    encoder = eval_encoder(phi, None, None, 3, 3)
    refs = [StateRef("data/sampling#000000", 2), StateRef("data/sampling#000000", 0)]
    with torch.no_grad():
        result = encoder(refs)
    assert calls == refs
    assert phi.training
    torch.testing.assert_close(
        result[0], torch.tensor([[[4.0, 1.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result[1], torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    )
    assert all(not value.requires_grad for value in result)
