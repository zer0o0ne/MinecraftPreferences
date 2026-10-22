"""`LayerS`, `LayerX` of src/mcpref/preferences/layers.py against specs/preference.md (5), (6)
and Interface (the constructor of `Layer`, `causal` false): a direct float64 composition with
the layers' own weights (preference_oracle.py). The tests of (2)-(4) live in
tests/transformer/test_layers.py."""

import pytest
import torch
from torch.testing import assert_close

from mcpref.preferences.layers import LayerS, LayerX
from mcpref.transformer.layers import Layer

from .preference_oracle import (
    cfg_of,
    layer_s,
    layer_x,
    positions,
    randomize_gains,
    weights,
)

F64 = torch.float64


def make(cls, cfg=None, seed: int = 0):
    """A layer in float64 with random weights and gains, built as `Preference` builds it."""
    cfg = cfg or cfg_of()
    torch.manual_seed(seed)
    layer = cls(cfg.width, cfg.hidden, cfg.heads, cfg.rope_base, cfg.norm_eps).double()
    randomize_gains(layer)
    return layer


@pytest.mark.parametrize("cls", [LayerS, LayerX])
def test_kinds_of_layer(cls):
    """Both kinds are a `Layer`, with its parameters in its attribute order."""
    layer = make(cls)
    assert isinstance(layer, Layer)
    names = [n for n, _ in layer.named_parameters()]
    assert names == [f"{k}.weight" for k in ("attn_norm", "q", "k", "v", "o", "mlp_norm",
                                             "w1", "w3", "w2")]  # fmt: skip


def inputs(n_a: int, n_b: int, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    A, B = torch.randn(2, n_a, 8, generator=g, dtype=F64), torch.randn(2, n_b, 8, generator=g, dtype=F64)  # fmt: skip
    return A, B, positions(n_a), positions(n_b)


@pytest.mark.parametrize("n_a, n_b", [(3, 3), (2, 5), (1, 4)])
def test_layer_s_against_composition(n_a, n_b):
    """(5): each argument alone, the same weights for both."""
    layer, cfg = make(LayerS), cfg_of()
    A, B, pa, pb = inputs(n_a, n_b, 8)
    got_a, got_b = layer(A, B, pa, pb)
    W = weights(layer)
    assert_close(got_a, layer_s(W, A, pa, cfg))
    assert_close(got_b, layer_s(W, B, pb, cfg))


def test_layer_s_arguments_do_not_meet():
    """(5): A's output does not depend on B."""
    layer = make(LayerS)
    A, B, pa, pb = inputs(3, 2, 9)
    B2 = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(10), dtype=F64)
    assert_close(layer(A, B, pa, pb)[0], layer(A, B2, pa, positions(4))[0])


@pytest.mark.parametrize("n_a, n_b", [(3, 3), (2, 5), (1, 4)])
def test_layer_x_against_composition(n_a, n_b):
    """(6): both attentions from the layer's inputs, one set of weights."""
    layer, cfg = make(LayerX), cfg_of()
    A, B, pa, pb = inputs(n_a, n_b, 11)
    got = layer(A, B, pa, pb)
    want = layer_x(weights(layer), A, B, pa, pb, cfg)
    assert_close(got[0], want[0])
    assert_close(got[1], want[1])


def test_layer_x_swapped_arguments():
    """(6) treats A and B alike: swapping the inputs swaps the outputs."""
    layer = make(LayerX)
    A, B, pa, pb = inputs(2, 4, 12)
    a, b = layer(A, B, pa, pb)
    b2, a2 = layer(B, A, pb, pa)
    assert_close(a, a2)
    assert_close(b, b2)


def test_not_causal():
    """`causal` false: in both kinds the first row sees the later rows of its keys."""
    for cls in (LayerS, LayerX):
        layer = make(cls)
        A, B, pa, pb = inputs(3, 3, 13)
        A2, B2 = A.clone(), B.clone()
        A2[:, 2:] += 1.0
        B2[:, 2:] += 1.0
        assert not torch.allclose(
            layer(A, B, pa, pb)[0][:, 0], layer(A2, B2, pa, pb)[0][:, 0]
        )
