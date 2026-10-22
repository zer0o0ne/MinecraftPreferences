"""`Policy`, `Dynamics`, `LayerC` of src/mcpref/policy/network.py against specs/policy.md (10),
(11) and Interface: the attribute names (`state_dict()` keys) and kinds, the shapes, no
buffers, the initialisation under its own seed, the groups."""

import dataclasses

import torch
from torch import nn

from mcpref.policy.network import Dynamics, LayerC, Policy
from mcpref.transformer.layers import Layer

from .policy_oracle import cfg_of
from .policy_world import NAMES

D, A = 5, len(NAMES)
PER_LAYER = ["attn_norm", "q", "k", "v", "o", "mlp_norm", "w1", "w3", "w2"]


def keys(cfg) -> list[str]:
    """state_dict() keys in the attribute order of Interface."""
    layers = [f"layers.{i}.{k}.weight" for i in range(cfg.layers) for k in PER_LAYER]
    return ["embed.weight", "executed.weight", *layers, "norm.weight", "head.weight",
            "actions.weight", "dynamics.norm.weight", "dynamics.w1.weight",
            "dynamics.w3.weight", "dynamics.w2.weight"]  # fmt: skip


def test_state_dict_keys_shapes_and_no_buffers():
    cfg = cfg_of(layers=3)
    net = Policy(cfg, D, NAMES)
    d, f, g = cfg.width, cfg.hidden, cfg.shift_hidden
    shapes = {"embed.weight": (d, D), "executed.weight": (d, A + 1), "norm.weight": (d,),
              "head.weight": (A, d), "actions.weight": (A, d),
              "dynamics.norm.weight": (2 * d,), "dynamics.w1.weight": (g, 2 * d),
              "dynamics.w3.weight": (g, 2 * d), "dynamics.w2.weight": (D, g)}  # fmt: skip
    layer = {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "w1": (f, d), "w3": (f, d),
             "w2": (d, f), "attn_norm": (d,), "mlp_norm": (d,)}  # fmt: skip
    state = net.state_dict()
    assert list(state) == keys(cfg)
    for name, value in state.items():
        want = layer[name.split(".")[2]] if name.startswith("layers.") else shapes[name]
        assert value.shape == want and value.dtype == torch.float32, name
    assert list(net.buffers()) == []


def test_module_kinds():
    """Linears without bias; norms RMSNorm(d or 2d, eps); E an Embedding with torch's
    arguments; layers a ModuleList of LayerC (each a Layer); dynamics a Dynamics."""
    cfg = cfg_of(layers=2, norm_eps=0.125)
    net = Policy(cfg, D, NAMES)
    assert type(net.layers) is nn.ModuleList and len(net.layers) == 2
    assert all(type(x) is LayerC and isinstance(x, Layer) for x in net.layers)
    assert type(net.dynamics) is Dynamics
    dyn = net.dynamics
    linears = [net.embed, net.executed, net.head, dyn.w1, dyn.w3, dyn.w2]
    linears += [getattr(x, k) for x in net.layers for k in ("q", "k", "v", "o", "w1", "w3", "w2")]  # fmt: skip
    assert all(type(x) is nn.Linear and x.bias is None for x in linears)
    norms = [net.norm, dyn.norm] + [getattr(x, k) for x in net.layers for k in ("attn_norm", "mlp_norm")]  # fmt: skip
    for x in norms:
        assert type(x) is nn.RMSNorm and x.eps == 0.125 and x.elementwise_affine
    assert tuple(dyn.norm.normalized_shape) == (2 * cfg.width,)
    E = net.actions
    assert type(E) is nn.Embedding and (E.num_embeddings, E.embedding_dim) == (
        A,
        cfg.width,
    )
    assert E.padding_idx is None and E.max_norm is None
    assert E.scale_grad_by_freq is False and E.sparse is False


def test_dynamics_constructor():
    """Dynamics(width = d, hidden, latent, eps): RMSNorm(2d), w1, w3 Linear(2d, hidden),
    w2 Linear(hidden, latent), all without bias, in this order."""
    dyn = Dynamics(4, 7, 3, 0.5)
    names = [(n, tuple(p.shape)) for n, p in dyn.named_parameters()]
    assert names == [("norm.weight", (8,)), ("w1.weight", (7, 8)), ("w3.weight", (7, 8)),
                     ("w2.weight", (3, 7))]  # fmt: skip
    assert dyn.norm.eps == 0.5 and all(x.bias is None for x in (dyn.w1, dyn.w3, dyn.w2))


def test_initialisation_by_hand():
    """(10): after manual_seed(seed) the submodules drawn by torch's defaults in Interface
    order (Linears U(+-1/sqrt(fan_in)), W_c too; E N(0, 1); RMSNorm draws nothing), every
    gain 1."""
    cfg = cfg_of(layers=2, seed=123)
    net = Policy(cfg, D, NAMES)
    d, f, g = cfg.width, cfg.hidden, cfg.shift_hidden
    lin = lambda n_in, n_out: nn.Linear(n_in, n_out, bias=False).weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        want = {"embed.weight": lin(D, d), "executed.weight": lin(A + 1, d)}
        for i in range(cfg.layers):
            for k, (n_in, n_out) in zip(
                ("q", "k", "v", "o", "w1", "w3", "w2"),
                [(d, d)] * 4 + [(d, f), (d, f), (f, d)],
            ):
                want[f"layers.{i}.{k}.weight"] = lin(n_in, n_out)
        want["head.weight"] = lin(d, A)
        want["actions.weight"] = nn.Embedding(A, d).weight
        want["dynamics.w1.weight"] = lin(2 * d, g)
        want["dynamics.w3.weight"] = lin(2 * d, g)
        want["dynamics.w2.weight"] = lin(g, D)
    state = net.state_dict()
    for name, value in want.items():
        assert torch.equal(state[name], value.detach()), name
    for name in set(state) - set(want):
        assert torch.equal(state[name], torch.ones_like(state[name])), name


def test_same_seed_same_weights_global_generator_untouched():
    """(10): equal seeds give equal weights whatever the global state; the global CPU
    generator's state is unchanged; another seed gives other weights."""
    cfg = cfg_of()
    torch.manual_seed(1)
    before = torch.get_rng_state()
    first = Policy(cfg, D, NAMES).state_dict()
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(2)
    second = Policy(cfg, D, NAMES).state_dict()
    other = Policy(dataclasses.replace(cfg, seed=12), D, NAMES).state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(first["embed.weight"], other["embed.weight"])
    assert not torch.equal(first["actions.weight"], other["actions.weight"])


def test_groups():
    """(11): pi (the weights of the Linears and E) then pi_no_decay (the gains), by
    identity in named_parameters() order; together each parameter once."""
    cfg = cfg_of(layers=2)
    net = Policy(cfg, D, NAMES)
    groups = net.groups()
    assert list(groups) == ["pi", "pi_no_decay"]
    named = list(net.named_parameters())
    gain = lambda n: n.endswith("norm.weight")
    want = {"pi": [p for n, p in named if not gain(n)],
            "pi_no_decay": [p for n, p in named if gain(n)]}  # fmt: skip
    assert len(want["pi_no_decay"]) == 2 * cfg.layers + 2
    for q, ps in want.items():
        assert len(groups[q]) == len(ps) and all(a is b for a, b in zip(groups[q], ps))
