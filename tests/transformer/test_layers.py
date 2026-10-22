"""`rope`, `split_heads`, `attention`, `mlp`, `rms_norm`, `Layer` of
src/mcpref/transformer/layers.py against specs/preference.md (2)-(4) and Interface, with
`causal` of specs/policy.md (3): hand vectors, a naive float64 loop, a direct float64
composition with the layer's own weights (layers_oracle.py). Moved from
tests/preferences/test_layers.py."""

import math

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from mcpref.transformer.layers import (
    Layer,
    attention,
    mlp,
    rms_norm,
    rope,
    split_heads,
)

from .layers_oracle import (
    ATTRIBUTES,
    attn_loop,
    norm,
    positions,
    randomize_gains,
    rope_one,
    update,
    weights,
)
from .layers_oracle import mlp as mlp_oracle

F64 = torch.float64
WIDTH, HIDDEN, HEADS, BASE, EPS = 8, 12, 2, 7.0, 0.05  # d_h = 4: two RoPE frequencies


def make(seed: int = 0) -> Layer:
    """A layer in float64 with random weights and gains."""
    torch.manual_seed(seed)
    layer = Layer(WIDTH, HIDDEN, HEADS, BASE, EPS).double()
    randomize_gains(layer)
    return layer


def randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def test_rope_hand_vector():
    """d_h = 4, base 100: theta = (1, 0.1); position 2 rotates (x0, x2) by 2, (x1, x3) by 0.2."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=F64)
    c1, s1, c2, s2 = math.cos(2), math.sin(2), math.cos(0.2), math.sin(0.2)
    want = [1 * c1 - 3 * s1, 2 * c2 - 4 * s2, 3 * c1 + 1 * s1, 4 * c2 + 2 * s2]
    got = rope(x, torch.tensor([2.0], dtype=F64), 100.0)
    assert_close(got, torch.tensor([want], dtype=F64))


def test_rope_against_transcription_with_leading_axes():
    """(…, n, d_h) with positions (n,) broadcast over the leading axes; d_h = 6."""
    x = randn(2, 3, 4, 6, seed=1)
    pos = torch.tensor([-3.0, -1.0, 0.0, 2.5], dtype=F64)
    got = rope(x, pos, 9.0)
    want = [[[rope_one(x[a, b, j].tolist(), float(pos[j]), 9.0) for j in range(4)]
             for b in range(3)] for a in range(2)]  # fmt: skip
    assert got.shape == x.shape
    assert_close(got, torch.tensor(want, dtype=F64))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.float32])
def test_rope_positions_of_any_dtype(dtype):
    """Positions of an integer or real dtype, cast to the dtype of x."""
    x = randn(2, 3, 4, seed=13)
    want = rope(x, torch.tensor([-2.0, -1.0, 0.0], dtype=F64), 10.0)
    assert_close(rope(x, torch.tensor([-2, -1, 0], dtype=dtype), 10.0), want)


def test_rope_position_zero_is_the_identity():
    x = randn(5, 8, seed=2)
    assert_close(rope(x, torch.zeros(5, dtype=F64), 10000.0), x)


def test_rope_scores_depend_only_on_the_offset():
    """R(q, a) . R(kappa, b) depends on a - b alone."""
    g = torch.Generator().manual_seed(3)
    q, k = (torch.randn(1, 8, generator=g, dtype=F64) for _ in range(2))

    def score(a: float, b: float) -> torch.Tensor:
        ra = rope(q, torch.tensor([a], dtype=F64), 10.0)
        return (ra * rope(k, torch.tensor([b], dtype=F64), 10.0)).sum()

    for a, b in [(5.0, 3.0), (-1.0, -3.0), (2.0, 0.0)]:
        assert_close(score(a, b), score(0.0, -2.0))
    assert not torch.allclose(score(3.0, 2.0), score(0.0, -2.0))


def test_rope_keeps_float32():
    """(2) in the dtype of x: float32 in, float32 out, close to the float64 transcription."""
    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(4))
    pos = torch.tensor([-2.0, -1.0, 0.0])
    got = rope(x, pos, 10000.0)
    want = [rope_one(x[j].double().tolist(), float(pos[j]), 10000.0) for j in range(3)]
    assert got.dtype == torch.float32
    assert_close(got, torch.tensor(want, dtype=torch.float32), rtol=1e-5, atol=1e-6)


def test_split_heads_by_hand():
    """(m, n, d) -> (m, h, n, d_h): head eta the columns (eta - 1) d_h .. eta d_h - 1."""
    x = torch.arange(2 * 2 * 6, dtype=F64).reshape(2, 2, 6)
    got = split_heads(x, 3)
    assert got.shape == (2, 3, 2, 2)
    for e in range(3):
        assert torch.equal(got[:, e], x[:, :, 2 * e : 2 * e + 2])


@pytest.mark.parametrize("causal", [False, True])
def test_attention_against_naive_loop(causal):
    """(3) per pair, head, row, n_q != n_k, positions not aligned; causal: j' <= j only."""
    layer = make()
    Q, K = randn(2, 3, 8, seed=5), randn(2, 5, 8, seed=6)
    pq = torch.tensor([-4.0, -1.0, 0.0], dtype=F64)
    pk = torch.tensor([-4.0, -3.0, -2.0, -1.0, 3.0], dtype=F64)
    got = attention(layer, Q, K, pq, pk, 2, 7.0, causal)
    assert got.shape == (2, 3, 8)
    assert_close(got, attn_loop(weights(layer), Q, K, pq, pk, 2, 7.0, causal))


def test_causal_self_attention_against_naive_loop():
    """Causal, n_q = n_k (the policy's case), positions varpi."""
    layer = make(1)
    X, pos = randn(2, 4, 8, seed=7), positions(4)
    got = attention(layer, X, X, pos, pos, 2, 7.0, True)
    assert_close(got, attn_loop(weights(layer), X, X, pos, pos, 2, 7.0, True))


def test_causal_rows_ignore_later_keys():
    """Causal: row j unchanged when keys j' > j change; not causal: every row changes."""
    layer = make(2)
    Q, K, pos = randn(1, 4, 8, seed=8), randn(1, 4, 8, seed=9), positions(4)
    K2 = K.clone()
    K2[:, 2:] += 1.0
    a = attention(layer, Q, K, pos, pos, 2, 7.0, True)
    b = attention(layer, Q, K2, pos, pos, 2, 7.0, True)
    assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:])
    full = attention(layer, Q, K, pos, pos, 2, 7.0, False)
    full2 = attention(layer, Q, K2, pos, pos, 2, 7.0, False)
    assert not torch.allclose(full[:, 0], full2[:, 0])


def test_attention_heads_and_base_as_given():
    """The same layer with heads = 4 (d_h = 2) and another base against the loop."""
    layer = make()
    Q, K = randn(1, 4, 8, seed=6), randn(1, 2, 8, seed=16)
    pq, pk = positions(4), positions(2)
    got = attention(layer, Q, K, pq, pk, 4, 3.0, False)
    assert_close(got, attn_loop(weights(layer), Q, K, pq, pk, 4, 3.0))


def test_mlp_against_transcription():
    layer = make()
    x = randn(2, 3, 8, seed=7)
    assert_close(mlp(layer, x), mlp_oracle(weights(layer), x))


def test_rms_norm():
    """N: torch.nn.RMSNorm(width, eps) with an affine gain of ones."""
    module = rms_norm(6, 0.125)
    assert type(module) is nn.RMSNorm and module.eps == 0.125
    assert tuple(module.normalized_shape) == (6,) and module.elementwise_affine
    assert torch.equal(module.weight, torch.ones(6))
    x = randn(3, 6, seed=10)
    assert_close(module.double()(x), norm(x, torch.ones(6, dtype=F64), 0.125))


def test_layer_attributes_in_order():
    """The parameters created in the attribute order of Interface, with their shapes;
    Linears without bias; norms RMSNorm(d, eps); no buffers."""
    layer = Layer(WIDTH, HIDDEN, HEADS, BASE, 0.25)
    d, f = WIDTH, HIDDEN
    shapes = {"q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d), "w1": (f, d),
              "w3": (f, d), "w2": (d, f), "attn_norm": (d,), "mlp_norm": (d,)}  # fmt: skip
    names = [n for n, _ in layer.named_parameters()]
    assert names == [f"{k}.weight" for k in ATTRIBUTES]
    for k in ATTRIBUTES:
        module = getattr(layer, k)
        assert module.weight.shape == shapes[k], k
        if k.endswith("norm"):
            assert type(module) is nn.RMSNorm and module.eps == 0.25
        else:
            assert type(module) is nn.Linear and module.bias is None
    assert list(layer.buffers()) == []


def test_layer_draws_its_linears_in_order():
    """torch's defaults in attribute order from the global generator (RMSNorm draws none)."""
    torch.manual_seed(3)
    layer = Layer(WIDTH, HIDDEN, HEADS, BASE, EPS)
    torch.manual_seed(3)
    sizes = [(WIDTH, WIDTH)] * 4 + [(WIDTH, HIDDEN), (WIDTH, HIDDEN), (HIDDEN, WIDTH)]
    for k, (n_in, n_out) in zip(("q", "k", "v", "o", "w1", "w3", "w2"), sizes):
        want = nn.Linear(n_in, n_out, bias=False).weight
        assert torch.equal(getattr(layer, k).weight, want), k


@pytest.mark.parametrize("causal", [False, True])
def test_layer_update_against_composition(causal):
    """`update`: H + Attn(H_hat, G_hat) with the layer's heads and base, then
    + F(N_gamma2(.)), with the layer's eps; queries of another length than the keys."""
    layer = make(4)
    H, H_hat, G_hat = randn(2, 3, 8, seed=11), randn(2, 3, 8, seed=12), randn(2, 5, 8, seed=14)  # fmt: skip
    ph, pg = positions(3), torch.tensor([-6.0, -2.0, -1.0, 0.0, 1.0], dtype=F64)
    got = layer.update(H, H_hat, G_hat, ph, pg, causal)
    want = update(weights(layer), H, H_hat, G_hat, ph, pg, HEADS, BASE, EPS, causal)
    assert_close(got, want)
