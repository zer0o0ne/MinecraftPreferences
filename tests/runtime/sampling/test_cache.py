"""Real mixed/sliding DynamicCache continuation of specs/env-sampling.md Datasets."""

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5 import Qwen3_5Config

from mcpref.runtime.sampling.models import decode_stream, encode_stream
from mcpref.runtime.sampling.shards import load_state, write_state
from runtime.sampling.helpers import FakePhi, FakeStream
from runtime.sampling.test_shards import sampler_state

CACHE_FIELDS = {
    "keys",
    "values",
    "sliding",
    "cumulative_length",
    "conv_states",
    "recurrent_states",
}


def assert_mapping(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_mapping(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_mapping(a, b)
    elif isinstance(expected, torch.Tensor):
        assert actual.device.type == "cpu"
        assert torch.equal(actual, expected)
    else:
        assert actual == expected


def qwen_phi(layer_types, sliding_window=4):
    phi = FakePhi()
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 8,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "layer_types": layer_types,
            "sliding_window": sliding_window,
        }
    )
    phi.backbone.config = config.text_config
    return phi


def mixed_stream(phi):
    stream = FakeStream(phi, 3)
    stream.ids = [4, 8]
    stream.grids = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
    stream.first = {"id": 4, "nested": {"value": 1.5}}
    cache = DynamicCache(config=phi.backbone.config)
    conv = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4)
    recurrent = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3) + 30
    keys = torch.arange(6, dtype=torch.float64).reshape(1, 1, 2, 3) + 50
    values = torch.arange(6, dtype=torch.float64).reshape(1, 1, 2, 3) + 60
    cache.update_conv_state(conv, 0, state_idx=0, conv_kernel_size=4)
    cache.update_recurrent_state(recurrent, 0, state_idx=0)
    cache.update(keys, values, 1)
    stream.cache = cache
    return stream, (conv, recurrent, keys, values)


def assert_linear_flags(cache):
    layer = cache.layers[0]
    assert layer.is_conv_states_initialized == {0: True}
    assert layer.is_recurrent_states_initialized == {0: True}
    assert layer.has_previous_state == {0: True}


def test_mixed_cache_round_trip_flags_and_identical_next_updates():
    phi = qwen_phi(["linear_attention", "full_attention"])
    stream, originals = mixed_stream(phi)
    encoded = encode_stream(stream)
    assert len(encoded["cache"]) == 2
    assert all(set(entry) == CACHE_FIELDS for entry in encoded["cache"])
    linear, full = encoded["cache"]
    assert [linear[name] for name in ("keys", "values", "sliding")] == [None] * 3
    assert linear["cumulative_length"] is None
    assert [full[name] for name in ("conv_states", "recurrent_states")] == [None] * 2
    assert full["cumulative_length"] is None
    entries = (linear, linear, full, full)
    names = ("conv_states", "recurrent_states", "keys", "values")
    for entry, name, tensor in zip(entries, names, originals, strict=True):
        assert entry[name].dtype == torch.float64
        assert entry[name].data_ptr() != tensor.data_ptr()
    decoded = decode_stream(phi, 3, encoded, torch.device("cpu"))
    assert_linear_flags(decoded.cache)
    assert_mapping(encode_stream(decoded), encoded)

    next_conv = torch.arange(90, 96, dtype=torch.float64).reshape(1, 2, 3, 1)
    next_recurrent = torch.arange(12, dtype=torch.float64).reshape(1, 2, 2, 3) + 100
    next_key = torch.tensor([[[[120.0, 121.0, 122.0]]]], dtype=torch.float64)
    next_value = torch.tensor([[[[130.0, 131.0, 132.0]]]], dtype=torch.float64)
    for cache in (stream.cache, decoded.cache):
        full_conv = cache.update_conv_state(
            next_conv, 0, state_idx=0, conv_kernel_size=4
        )
        full_recurrent = cache.update_recurrent_state(next_recurrent, 0, state_idx=0)
        full_key, full_value = cache.update(next_key, next_value, 1)
        if cache is stream.cache:
            expected = (full_conv, full_recurrent, full_key, full_value)
        else:
            for actual, wanted in zip(
                (full_conv, full_recurrent, full_key, full_value), expected, strict=True
            ):
                assert torch.equal(actual, wanted)
    assert_mapping(encode_stream(decoded), encode_stream(stream))


def test_sliding_six_tokens_window_four_round_trip_masks_and_next_update():
    phi = qwen_phi(["sliding_attention"], sliding_window=4)
    stream = FakeStream(phi, 2)
    keys = torch.arange(6, dtype=torch.float64).reshape(1, 1, 6, 1)
    values = keys + 100
    stream.cache.update(keys, values, 0)
    assert stream.cache.layers[0].cumulative_length == 6
    assert torch.equal(stream.cache.layers[0].keys, keys[..., 3:, :])
    encoded = encode_stream(stream)
    assert encoded["cache"][0]["cumulative_length"] == 6
    decoded = decode_stream(phi, 2, encoded, torch.device("cpu"))
    assert decoded.cache.layers[0].cumulative_length == 6
    assert decoded.cache.layers[0].get_mask_sizes(2) == (5, 3)
    assert_mapping(encode_stream(decoded), encoded)

    next_keys = torch.tensor([[[[6.0], [7.0]]]], dtype=torch.float64)
    next_values = next_keys + 100
    expected = stream.cache.update(next_keys, next_values, 0)
    actual = decoded.cache.update(next_keys, next_values, 0)
    for got, wanted in zip(actual, expected, strict=True):
        assert torch.equal(got, wanted)
        assert got.shape[-2] == 5
    for cache in (stream.cache, decoded.cache):
        layer = cache.layers[0]
        assert layer.cumulative_length == 8
        assert layer.get_mask_sizes(2) == (5, 5)
        assert torch.equal(
            layer.keys, torch.tensor([[[[5.0], [6.0], [7.0]]]], dtype=torch.float64)
        )
    assert_mapping(encode_stream(decoded), encode_stream(stream))


def test_complete_mixed_cache_state_file_loads_weights_only(tmp_path, monkeypatch):
    phi = qwen_phi(["linear_attention", "full_attention"])
    original = sampler_state(FakePhi())
    original.stream = mixed_stream(phi)[0]
    out = tmp_path / "out"
    shard = out / "shards/000000"
    shard.mkdir(parents=True)
    write_state(shard, original)
    raw = torch.load(shard / "state.pt", map_location="cpu", weights_only=True)
    loads = []
    real_load = torch.load

    def load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("mcpref.runtime.sampling.shards.torch.load", load)
    restored = load_state(out, phi, 3, torch.device("cpu"))
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert_mapping(encode_stream(restored.stream), raw["stream"])
    assert restored.env == original.env
    assert_linear_flags(restored.stream.cache)
