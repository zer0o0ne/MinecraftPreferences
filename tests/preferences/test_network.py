"""`Preference.forward` of src/mcpref/preferences/network.py against specs/preference.md (1),
(7), (8) (a direct float64 composition with the module's own weights), the used rows
$n = \\min(L_P, \\ell)$ and their positions, a list without X, the swap (9), no dropout,
float32."""

import pytest
import torch
from torch.testing import assert_close

from mcpref.preferences.layers import LayerS, LayerX
from mcpref.preferences.network import Preference

from .preference_oracle import cfg_of, preference, randomize_gains

F64, D = torch.float64, 5


def net_of(cfg=None) -> Preference:
    """P in float64 with every gain and b_inc away from its initial value."""
    net = Preference(cfg or cfg_of(), D).double()
    randomize_gains(net)
    return net


def histories(m: int, l_a: int, l_b: int, seed: int, dtype=F64) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, l_a, D, generator=g, dtype=dtype),
            torch.randn(m, l_b, D, generator=g, dtype=dtype))  # fmt: skip


def assert_oracle(net, cfg, Ya, Yb) -> tuple:
    s, t = net(Ya, Yb)
    want_s, want_t = preference(net, cfg, Ya, Yb)
    assert s.shape == t.shape == (Ya.shape[0],)
    assert_close(s, want_s)
    assert_close(t, want_t)
    return s, t


LAYERS = {"S X": ("S", "X"), "S X S X": ("S", "X", "S", "X"), "X": ("X",),
          "S": ("S",), "X X S": ("X", "X", "S")}  # fmt: skip


@pytest.mark.parametrize("layers", LAYERS.values(), ids=LAYERS)
def test_forward_against_composition(layers):
    """(1), (7), (8) with the layers in the order of Lambda, each of its kind."""
    cfg = cfg_of(layers=layers)
    net = net_of(cfg)
    assert isinstance(net.layers, torch.nn.ModuleList) and len(net.layers) == len(
        layers
    )
    kinds = {"S": LayerS, "X": LayerX}
    assert all(type(x) is kinds[k] for x, k in zip(net.layers, layers))
    assert_oracle(net, cfg, *histories(3, 3, 3, 1))


HISTORIES = {  # L_P = 3: lengths l_a, l_b
    "full": (3, 3),
    "a shorter": (2, 3),
    "both shorter, unequal": (1, 2),
    "a longer": (6, 3),
    "both longer, unequal": (4, 7),
    "shorter and longer": (2, 5),
}


@pytest.mark.parametrize("l_a, l_b", HISTORIES.values(), ids=HISTORIES)
def test_used_rows_and_positions(l_a, l_b):
    """n = min(L_P, l) rows ending at the compared state, positions aligned at 0."""
    cfg = cfg_of()
    net = net_of(cfg)
    Ya, Yb = histories(2, l_a, l_b, 2)
    s, _ = assert_oracle(net, cfg, Ya, Yb)
    if min(3, l_a) != min(3, l_b):  # n_a != n_b: positions 0..n-1 give other logits
        other = preference(net, cfg, Ya, Yb, lambda n: torch.arange(n, dtype=F64))[0]
        assert not torch.allclose(s, other)


def test_older_rows_unused():
    """Rows before the last L_P change nothing; the output equals that on the last rows."""
    net = net_of()
    Ya, Yb = histories(2, 6, 5, 3)
    Ya2, Yb2 = Ya.clone(), Yb.clone()
    Ya2[:, :3], Yb2[:, :2] = 100.0, -7.0
    for got, want in zip(net(Ya2, Yb2), net(Ya, Yb)):
        assert_close(got, want)
    for got, want in zip(net(Ya[:, 3:], Yb[:, 2:]), net(Ya, Yb)):
        assert_close(got, want)


def test_one_pair_and_one_row():
    """m = 1 gives shape (1,); l = 1: the compared state alone, at position 0."""
    cfg = cfg_of()
    assert_oracle(net_of(cfg), cfg, *histories(1, 1, 4, 4))


def test_pairs_independent():
    """Entry r depends only on pair r."""
    net = net_of()
    Ya, Yb = histories(3, 3, 2, 5)
    Ya2 = Ya.clone()
    Ya2[1] = 3.0
    s, t = net(Ya, Yb)
    s2, t2 = net(Ya2, Yb)
    assert_close(s2[[0, 2]], s[[0, 2]])
    assert_close(t2[[0, 2]], t[[0, 2]])
    assert not torch.allclose(s2[1], s[1])


def test_swap():
    """(9) on random weights: P(Y_b, Y_a) = (-s, t), also for l_a != l_b; s(Y, Y) = 0."""
    net = net_of(cfg_of(layers=("X", "S", "X")))
    Ya, Yb = histories(4, 2, 5, 6)
    s, t = net(Ya, Yb)
    s2, t2 = net(Yb, Ya)
    assert_close(s2, -s)
    assert_close(t2, t)
    assert_close(net(Ya, Ya)[0], torch.zeros(4, dtype=F64), atol=1e-12, rtol=0)


def test_without_x_a_difference_of_one_function():
    """Lambda without X: s(a, b) = g(a) - g(b), so s(a, b) = s(a, c) - s(b, c), and
    t(a, b) - t(a, c) = t(b, b') - t(c, b') (t = h(a) + h(b) + b_inc)."""
    net = net_of(cfg_of(layers=("S", "S")))
    a, b = histories(2, 3, 4, 7)
    c, e = histories(2, 2, 5, 8)
    s_ab, t_ab = net(a, b)
    s_ac, t_ac = net(a, c)
    s_bc = net(b, c)[0]
    assert_close(s_ab, s_ac - s_bc)
    assert_close(t_ab - t_ac, net(b, e)[1] - net(c, e)[1])


def test_with_x_not_a_difference():
    """Sanity of the previous test: with an X layer the identity fails."""
    net = net_of(cfg_of(layers=("X",)))
    a, b = histories(2, 3, 3, 7)
    c, _ = histories(2, 3, 3, 8)
    assert not torch.allclose(net(a, b)[0], net(a, c)[0] - net(b, c)[0])


def test_no_dropout():
    """Train and eval modes give the same logits; repeated calls agree."""
    net = net_of()
    Ya, Yb = histories(2, 3, 3, 9)
    net.train()
    first, second = net(Ya, Yb), net(Ya, Yb)
    net.eval()
    third = net(Ya, Yb)
    for k in range(2):
        assert torch.equal(first[k], second[k])
        assert_close(third[k], first[k])


def test_float32():
    """Parameters and inputs float32 (the Domain): logits float32, close to the oracle."""
    cfg = cfg_of()
    net = Preference(cfg, D)
    randomize_gains(net)
    Ya, Yb = histories(2, 2, 4, 10, torch.float32)
    s, t = net(Ya, Yb)
    assert s.dtype == t.dtype == torch.float32
    want = preference(net.double(), cfg, Ya.double(), Yb.double())
    assert_close(s.double(), want[0], rtol=1e-4, atol=1e-5)
    assert_close(t.double(), want[1], rtol=1e-4, atol=1e-5)
