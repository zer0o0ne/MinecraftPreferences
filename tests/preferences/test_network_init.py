"""`Preference` of src/mcpref/preferences/network.py against specs/preference.md (11), (12)
and Interface: the initialisation under its own seed, the groups, the attribute names
(`state_dict()` keys), the modules' kinds and shapes, no buffers."""

import dataclasses

import torch
from torch import nn

from mcpref.preferences.network import Preference

from .preference_oracle import GAINS, LINEARS, cfg_of, decayed_names

D = 5
LAYERS = ("S", "X", "X")


def keys(cfg) -> list[str]:
    """state_dict() keys in the attribute order of Interface."""
    per_layer = ["attn_norm", "q", "k", "v", "o", "mlp_norm", "w1", "w3", "w2"]
    layers = [
        f"layers.{i}.{k}.weight" for i in range(len(cfg.layers)) for k in per_layer
    ]
    return ["embed.weight", *layers, "norm.weight", "pref.weight", "inc.weight", "inc.bias"]  # fmt: skip


def test_state_dict_keys_shapes_and_no_buffers():
    cfg = cfg_of(layers=LAYERS)
    net = Preference(cfg, D)
    d, f = cfg.width, cfg.hidden
    shapes = {"embed": (d, D), "q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d),
              "w1": (f, d), "w3": (f, d), "w2": (d, f), "norm": (d,),
              "pref": (1, d), "inc": (1, d)}  # fmt: skip
    state = net.state_dict()
    assert list(state) == keys(cfg)
    for name, value in state.items():
        part = name.split(".")[-2]
        want = (1,) if name == "inc.bias" else shapes.get(part, (d,))
        assert value.shape == want, name
        assert value.dtype == torch.float32, name
    assert list(net.buffers()) == []


def test_module_kinds():
    """Linears without bias except `inc`; norms RMSNorm(d, eps) with a gain."""
    cfg = cfg_of(layers=LAYERS, norm_eps=0.125)
    net = Preference(cfg, D)
    linears = [net.embed, net.pref, net.inc] + [
        getattr(x, k) for x in net.layers for k in LINEARS
    ]
    assert all(type(x) is nn.Linear for x in linears)
    assert all(x.bias is None for x in linears if x is not net.inc)
    assert net.inc.bias is not None
    norms = [net.norm] + [getattr(x, k) for x in net.layers for k in GAINS]
    for x in norms:
        assert type(x) is nn.RMSNorm and x.eps == 0.125
        assert tuple(x.normalized_shape) == (cfg.width,) and x.weight is not None


def test_initialisation_by_hand():
    """(11): after manual_seed(seed) the Linears drawn by torch's defaults in Interface
    order (RMSNorm draws nothing), every gain 1."""
    cfg = cfg_of(layers=LAYERS, seed=123)
    net = Preference(cfg, D)
    d, f = cfg.width, cfg.hidden
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        want = {"embed.weight": nn.Linear(D, d, bias=False).weight}
        for i in range(len(LAYERS)):
            for k, (n_in, n_out) in zip(
                LINEARS, [(d, d)] * 4 + [(d, f), (d, f), (f, d)]
            ):
                want[f"layers.{i}.{k}.weight"] = nn.Linear(
                    n_in, n_out, bias=False
                ).weight
        want["pref.weight"] = nn.Linear(d, 1, bias=False).weight
        inc = nn.Linear(d, 1)
        want |= {"inc.weight": inc.weight, "inc.bias": inc.bias}
    state = net.state_dict()
    for name, value in want.items():
        assert torch.equal(state[name], value.detach()), name
    for name in set(state) - set(want):
        assert torch.equal(state[name], torch.ones(d)), name


def test_same_seed_same_weights_global_generator_untouched():
    """(11): equal seeds give equal weights whatever the global state; the global CPU
    generator's state is unchanged; another seed gives other weights."""
    cfg = cfg_of()
    torch.manual_seed(1)
    before = torch.get_rng_state()
    first = Preference(cfg, D).state_dict()
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(2)
    second = Preference(cfg, D).state_dict()
    other = Preference(dataclasses.replace(cfg, seed=12), D).state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(first["embed.weight"], other["embed.weight"])


def test_groups():
    """(12): P then P_no_decay, by identity in named_parameters() order; together each
    parameter once."""
    cfg = cfg_of(layers=LAYERS)
    net = Preference(cfg, D)
    groups = net.groups()
    assert list(groups) == ["P", "P_no_decay"]
    named = dict(net.named_parameters())
    decay, rest = decayed_names(cfg)
    order = list(named)
    for got, names in [(groups["P"], decay), (groups["P_no_decay"], rest)]:
        names = sorted(names, key=order.index)
        assert len(got) == len(names)
        assert all(p is named[n] for p, n in zip(got, names))
    ids = [id(p) for ps in groups.values() for p in ps]
    assert sorted(ids) == sorted(id(p) for p in net.parameters())
