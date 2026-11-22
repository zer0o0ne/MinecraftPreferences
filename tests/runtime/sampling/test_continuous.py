"""An endless world retains memory across history caps, shards and process resumes."""

import shutil

import pytest
import torch

from mcpref.runtime.sampling.models import decode_stream, encode_stream
from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import load_state, load_step
from runtime.sampling.continuous_world import modules, setup
from runtime.sampling.helpers import FakeEnv
from runtime.sampling.test_recovery import assert_record


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_world_memory_survives_history_and_shard_boundaries(site, monkeypatch):
    setup(site, monkeypatch)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    shapes = None
    for _ in range(6):
        sampler.sample()
        t = sampler.next_step
        expected = torch.tensor(
            [
                [j + (j + 1) * (j + 2) / 2, j + 0.5 + (j + 1) * (j + 2) / 2]
                for j in range(max(0, t - 4), t + 1)
            ],
            dtype=torch.float64,
        )[None]
        torch.testing.assert_close(sampler.history.Y, expected, rtol=0, atol=0)
        encoded = encode_stream(sampler.stream)
        assert set(encoded) == {"mode", "version", "origin", "layers"}
        assert encoded["origin"] == {"id": 0}
        current = [
            tuple(row[key].shape)
            for row in encoded["layers"]
            for key in ("conv", "recurrent")
        ]
        assert shapes is None or current == shapes
        shapes = current
    assert sampler.states == 5 and not site.length_checks
    assert [event[1] for event in site.events if event[0] == "stream"] == list(
        range(13)
    )
    assert not any(site.modules.phi.grad_modes)
    assert site.env.resets == 1


def test_continuous_resume_matches_uninterrupted_next_shard(site, monkeypatch):
    setup(site, monkeypatch)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    sampler.sample()
    sampler.sample()
    expected = load_state(site.out, sampler.modules.phi, 5, sampler.device)
    expected_records = [load_step(site.out, i) for i in (4, 5)]
    shutil.rmtree(site.out / "shards/000002")
    shutil.rmtree(site.root / "data/diagnostics")
    env = FakeEnv()
    restored = Sampler(site.cfg, env, modules())
    restored.reset()
    assert env.resets == 0 and env.restores == 1 and env.index == 4
    restored.sample()
    actual = load_state(site.out, restored.modules.phi, 5, restored.device)
    assert_tree_equal(encode_stream(actual.stream), encode_stream(expected.stream))
    assert_tree_equal(actual.history.Y, expected.history.Y)
    assert_tree_equal(actual.history.C, expected.history.C)
    assert actual.rng_state == expected.rng_state
    for expected_record in expected_records:
        assert_record(load_step(site.out, expected_record.index), expected_record)
    assert [
        event[1] for event in restored.modules.phi.events if event[0] == "stream"
    ] == [5, 6]


@pytest.mark.parametrize("change", ["weights", "mode", "version", "layers"])
def test_incompatible_resume_fails_before_touching_world(site, monkeypatch, change):
    setup(site, monkeypatch)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    if change != "weights":
        path = site.out / "shards/000000/state.pt"
        saved = torch.load(path, weights_only=True)
        if change == "mode":
            saved["stream"] = {"ids": [], "grids": [], "first": {}, "cache": []}
        elif change == "version":
            saved["stream"]["version"] = 999
        else:
            saved["stream"]["layers"] = []
        torch.save(saved, path)
    env = FakeEnv()
    restored = Sampler(
        site.cfg, env, modules("different" if change == "weights" else "fake-model")
    )
    with pytest.raises(
        RuntimeError, match="unchanged model weights|incompatible sampling state"
    ):
        restored.reset()
    assert env.resets == env.restores == 0 and not env.events


def test_stream_primitive_roundtrip_preserves_dtype_and_snapshot_independence(
    site, monkeypatch
):
    setup(site, monkeypatch)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    encoded = encode_stream(sampler.stream)
    path = site.root / "stream.pt"
    torch.save(encoded, path)
    stored = torch.load(path, weights_only=True)
    decoded = decode_stream(site.modules.phi, 5, stored, torch.device("cpu"))
    assert_tree_equal(encode_stream(decoded), encoded)
    assert decoded.state.layers[0].conv.dtype == torch.float32
    assert decoded.state.layers[0].recurrent.dtype == torch.float64
    sampler.stream.state.layers[0].recurrent.add_(10)
    assert decoded.state.layers[0].recurrent.item() == 6
    assert encoded["layers"][0]["recurrent"].item() == 6
