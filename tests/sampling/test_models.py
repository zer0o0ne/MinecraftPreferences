"""Pretrained/supplied modules and primitive stream state of specs/env-sampling.md (1)."""

import dataclasses
import hashlib

import torch
import yaml
from transformers import DynamicCache

from mcpref.sampling.models import (
    SamplingModules,
    decode_stream,
    encode_stream,
    load_pretrained,
)
from mcpref.sampling.runner import Sampler

from .helpers import FakeStream, modules


def test_sampling_modules_interface():
    assert [field.name for field in dataclasses.fields(SamplingModules)] == [
        "phi",
        "pi",
        "P",
        "fingerprint",
    ]


def test_pretrained_loading_eval_fingerprint_and_explicit_cpu_load(site, monkeypatch):
    source = modules("source")
    for module in (source.phi, source.pi, source.P):
        module.weight.data.fill_(3.25)
    source.phi.head[-1].weight.data.fill_(3.25)
    source.phi.head[-1].bias.data.fill_(-1.75)
    pretrain = yaml.safe_load((site.root / site.cfg.pretrain).read_text())
    model = site.root / pretrain["out_dir"] / "model.pt"
    torch.save(
        {name: getattr(source, name).state_dict() for name in ("phi", "pi", "P")},
        model,
    )
    built = modules("built")
    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return {"phi": built.phi, "pi": built.pi, "P": built.P}

    loads = []
    complete = []
    real_load = torch.load

    def load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)

    def finished_run(*args):
        complete.append(args)
        return {}

    monkeypatch.setattr("mcpref.sampling.models.build_modules", build)
    monkeypatch.setattr("mcpref.sampling.models.finished_run", finished_run)
    monkeypatch.setattr("mcpref.sampling.models.torch.load", load)
    got = load_pretrained(site.cfg)
    assert complete == [
        (site.root / pretrain["out_dir"], pretrain["out_dir"], "complete")
    ]
    assert calls[0][1] == {
        "paired": True,
        "device": "cpu",
        "latent_init_scale": pretrain["latent_init_scale"],
    }
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert got.fingerprint == hashlib.sha256(model.read_bytes()).hexdigest()
    for name in ("phi", "pi", "P"):
        module = getattr(got, name)
        assert not module.training
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 3.25))
    torch.testing.assert_close(
        got.phi.head[-1].weight, torch.full_like(got.phi.head[-1].weight, 3.25)
    )
    torch.testing.assert_close(
        got.phi.head[-1].bias, torch.full_like(got.phi.head[-1].bias, -1.75)
    )


def test_supplied_modules_eval_and_length_checks(site):
    for module in (site.modules.phi, site.modules.pi, site.modules.P):
        module.train()
    Sampler(site.cfg, site.env, site.modules)
    assert all(
        not module.training
        for module in (site.modules.phi, site.modules.pi, site.modules.P)
    )
    args, kwargs = site.length_checks[0]
    assert args[0] == 2
    assert kwargs == {"paired": True, "index_dir": site.cfg.perception_data}


def test_none_modules_delegates_to_pretrained_loader(site, monkeypatch):
    calls = []

    def load(cfg):
        calls.append(cfg)
        return site.modules

    monkeypatch.setattr("mcpref.sampling.runner.load_pretrained", load)
    Sampler(site.cfg, site.env)
    assert calls == [site.cfg]


def test_stream_round_trip_preserves_ids_grids_first_and_every_cache_entry(site):
    phi = site.modules.phi
    stream = FakeStream(phi, 3)
    stream.ids = [4, 8, 15]
    stream.grids = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
    stream.first = {"id": 4, "nested": {"x": 1.5}}
    key = lambda n: torch.arange(n, dtype=torch.float64).reshape(1, 1, 1, n)
    stream.cache = DynamicCache(
        ddp_cache_data=[
            (key(2), key(3), None),
            (key(4), key(5), torch.tensor(4)),
        ],
        config=phi.backbone.config,
    )
    encoded = encode_stream(stream)
    assert set(encoded) == {"ids", "grids", "cache", "first"}
    assert encoded["ids"] == [4, 8, 15] and encoded["first"] == stream.first
    assert all(tensor.device.type == "cpu" for tensor in encoded["grids"])
    assert encoded["cache"][0]["sliding"] is None
    torch.testing.assert_close(encoded["cache"][1]["sliding"], torch.tensor(4))
    decoded = decode_stream(phi, 3, encoded, torch.device("cpu"))
    assert decoded.states == 3 and decoded.first == stream.first
    again = encode_stream(decoded)
    assert again["ids"] == encoded["ids"] and again["first"] == encoded["first"]
    for actual, expected in zip(again["grids"], encoded["grids"], strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(again["cache"], encoded["cache"], strict=True):
        for name in ("keys", "values"):
            torch.testing.assert_close(actual[name], expected[name])
        if expected["sliding"] is None:
            assert actual["sliding"] is None
        else:
            torch.testing.assert_close(actual["sliding"], expected["sliding"])
