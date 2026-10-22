"""`rope`, `attention`, `mlp`, `LayerS`, `LayerX` of src/mcpref/preferences/layers.py against
specs/preference.md (2)-(6): hand vectors, a naive float64 loop, a direct float64
composition with the layers' own weights (preference_oracle.py)."""

import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.preferences.layers import LayerS, LayerX, attention, mlp, rope

from .preference_oracle import (
    attn_loop,
    cfg_of,
    layer_s,
    layer_x,
    positions,
    randomize_gains,
    rope_one,
    weights,
)
from .preference_oracle import mlp as mlp_oracle

F64 = torch.float64


def make(cls, cfg=None, seed: int = 0):
    """A layer in float64 with random weights and gains."""
    torch.manual_seed(seed)
    layer = cls(cfg or cfg_of()).double()
    randomize_gains(layer)
    return layer


def test_rope_hand_vector():
    """d_h = 4, base 100: theta = (1, 0.1); position 2 rotates (x0, x2) by 2, (x1, x3) by 0.2."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=F64)
    c1, s1, c2, s2 = math.cos(2), math.sin(2), math.cos(0.2), math.sin(0.2)
    want = [1 * c1 - 3 * s1, 2 * c2 - 4 * s2, 3 * c1 + 1 * s1, 4 * c2 + 2 * s2]
    got = rope(x, torch.tensor([2.0], dtype=F64), 100.0)
    assert_close(got, torch.tensor([want], dtype=F64))


def test_rope_against_transcription_with_leading_axes():
    """(…, n, d_h) with positions (n,) broadcast over the leading axes; d_h = 6."""
    x = torch.randn(2, 3, 4, 6, generator=torch.Generator().manual_seed(1), dtype=F64)
    pos = torch.tensor([-3.0, -1.0, 0.0, 2.5], dtype=F64)
    got = rope(x, pos, 9.0)
    want = [[[rope_one(x[a, b, j].tolist(), float(pos[j]), 9.0) for j in range(4)]
             for b in range(3)] for a in range(2)]  # fmt: skip
    assert got.shape == x.shape
    assert_close(got, torch.tensor(want, dtype=F64))


def test_rope_position_zero_is_the_identity():
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(2), dtype=F64)
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


@pytest.mark.parametrize("cls", [LayerS, LayerX])
def test_attention_against_naive_loop(cls):
    """(3) per pair, head, row, n_q != n_k, positions not aligned; no mask."""
    layer = make(cls)
    g = torch.Generator().manual_seed(5)
    Q, K = torch.randn(2, 3, 8, generator=g, dtype=F64), torch.randn(2, 5, 8, generator=g, dtype=F64)  # fmt: skip
    pq = torch.tensor([-4.0, -1.0, 0.0], dtype=F64)
    pk = torch.tensor([-4.0, -3.0, -2.0, -1.0, 3.0], dtype=F64)
    got = attention(layer, Q, K, pq, pk, 2, 7.0)
    assert got.shape == (2, 3, 8)
    assert_close(got, attn_loop(weights(layer), Q, K, pq, pk, 2, 7.0))


def test_attention_heads_and_base_as_given():
    """The same layer with heads = 4 (d_h = 2) and another base against the loop."""
    layer = make(LayerS)
    g = torch.Generator().manual_seed(6)
    Q, K = torch.randn(1, 4, 8, generator=g, dtype=F64), torch.randn(1, 2, 8, generator=g, dtype=F64)  # fmt: skip
    pq, pk = positions(4), positions(2)
    got = attention(layer, Q, K, pq, pk, 4, 3.0)
    assert_close(got, attn_loop(weights(layer), Q, K, pq, pk, 4, 3.0))


def test_mlp_against_transcription():
    layer = make(LayerX)
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(7), dtype=F64)
    assert_close(mlp(layer, x), mlp_oracle(weights(layer), x))


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
