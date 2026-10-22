"""`Policy.readout`, `logits`, `shift`, `forward`, `LayerC`, `Dynamics` of
src/mcpref/policy/network.py against specs/policy.md (2)-(7) without dropout (eval mode): a
naive float64 loop, a direct float64 composition with the module's own weights
(policy_oracle.py), the used rows and their positions, each row's readout equal to a call on
its prefix, a batch of actions, float32. The dropout of (2): test_network_dropout.py."""

import pytest
import torch
from torch.testing import assert_close
from transformer.layers_oracle import norm, positions, randomize_gains, update, weights

from mcpref.policy.network import Dynamics, LayerC, Policy

from . import policy_oracle as oracle
from .policy_oracle import cfg_of
from .policy_world import NAMES, make_clip

F64, D, A = torch.float64, 5, len(NAMES)


def net_of(cfg=None) -> Policy:
    """pi in float64, eval mode, every gain away from its initial value."""
    net = Policy(cfg or cfg_of(), D, NAMES).double().eval()
    randomize_gains(net)
    return net


def history(m: int, ell: int, seed: int, dtype=F64) -> tuple:
    """Y random; C rows random distributions over A and none (the last row none)."""
    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(m, ell, D, generator=g, dtype=dtype)
    C = torch.rand(m, ell, A + 1, generator=g, dtype=dtype) ** 3
    return Y, C / C.sum(-1, keepdim=True)


def test_readout_against_naive_loop():
    """(2)-(4), l = L_pi: the tokens, N_pi causal layers C by the naive loop, the norm."""
    cfg = cfg_of()
    net = net_of(cfg)
    Y, C = history(2, cfg.history, 1)
    got = net.readout(Y, C)
    assert got.shape == (2, cfg.history, cfg.width)
    assert_close(got, oracle.readout(net, cfg, Y, C, loop=True))


@pytest.mark.parametrize("ell", [1, 2, 4, 7], ids=lambda x: f"l {x}")
def test_used_rows_and_positions(ell):
    """n = min(L_pi, l) rows ending at the last, positions varpi_j = j - (n - 1)."""
    cfg = cfg_of()
    net = net_of(cfg)
    Y, C = history(3, ell, 2)
    got = net.readout(Y, C)
    assert got.shape == (3, min(cfg.history, ell), cfg.width)
    assert_close(got, oracle.readout(net, cfg, Y, C))


def test_older_rows_unused():
    """Rows before the last L_pi change nothing."""
    net = net_of()
    Y, C = history(2, 7, 3)
    Y2, C2 = Y.clone(), C.clone()
    Y2[:, :3] = 50.0
    C2[:, :3] = torch.nn.functional.one_hot(torch.tensor(2), A + 1).to(F64)
    assert_close(net.readout(Y2, C2), net.readout(Y, C))
    assert_close(net.readout(Y[:, 3:], C[:, 3:]), net.readout(Y, C))


@pytest.mark.parametrize("ell", [3, 4])
def test_each_row_equals_the_call_on_its_prefix(ell):
    """(4): row j (eval mode) equals the last row of a call on the rows 0..j."""
    net = net_of()
    Y, C = history(2, ell, 4)
    full = net.readout(Y, C)
    for j in range(ell):
        assert_close(full[:, j], net.readout(Y[:, : j + 1], C[:, : j + 1])[:, -1])


def test_executed_actions_enter():
    """(2): another C (the in-flight action of a row) changes that row and the later ones."""
    net = net_of()
    Y, C = history(1, 4, 5)
    C2 = C.clone()
    C2[:, 2] = torch.nn.functional.one_hot(torch.tensor(4), A + 1).to(F64)
    a, b = net.readout(Y, C), net.readout(Y, C2)
    assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_layer_c_against_composition():
    """(3): H + Attn_c(N(H), N(H)) with positions on both sides, then + F(N_2(.))."""
    cfg = cfg_of()
    torch.manual_seed(0)
    layer = LayerC(
        cfg.width, cfg.hidden, cfg.heads, cfg.rope_base, cfg.norm_eps
    ).double()
    randomize_gains(layer)
    H, pos = torch.randn(2, 4, cfg.width, dtype=F64), positions(4)
    W = weights(layer)
    x = norm(H, W["g1"], cfg.norm_eps)
    want = update(W, H, x, x, pos, pos, cfg.heads, cfg.rope_base, cfg.norm_eps, True)
    assert_close(layer(H, pos), want)


def test_dynamics_against_composition():
    """(6) on x = [r; e_a] of size 2d."""
    torch.manual_seed(0)
    dyn = Dynamics(4, 6, 3, 0.05).double()
    randomize_gains(dyn)
    x = torch.randn(2, 3, 8, dtype=F64)
    u = norm(x, dyn.norm.weight, 0.05)
    a = u @ dyn.w1.weight.T
    want = (a * torch.sigmoid(a) * (u @ dyn.w3.weight.T)) @ dyn.w2.weight.T
    assert_close(dyn(x), want)


@pytest.mark.parametrize(
    "lead", [(), (3,), (2, 3)], ids=["(d,)", "(3, d)", "(2, 3, d)"]
)
def test_logits(lead):
    """(5): r W_lambda, (..., d) -> (..., |A|), no bias."""
    net = net_of()
    R = torch.randn(*lead, net.head.in_features, dtype=F64)
    got = net.logits(R)
    assert got.shape == (*lead, A)
    assert_close(got, oracle.logits(net, R))
    assert torch.equal(net.logits(torch.zeros(net.head.in_features, dtype=F64)), torch.zeros(A, dtype=F64))  # fmt: skip


def test_shift_on_a_batch_of_actions():
    """(6): one action per readout, leading axes (2, 3); each entry its own composition."""
    cfg = cfg_of()
    net = net_of(cfg)
    R = torch.randn(2, 3, cfg.width, dtype=F64)
    a = torch.tensor([[0, 6, 2], [2, 2, 5]])
    got = net.shift(R, a)
    assert got.shape == (2, 3, D)
    assert_close(got, oracle.shift(net, cfg, R, a))
    for i in range(2):
        for j in range(3):
            assert_close(got[i, j], oracle.shift(net, cfg, R[i, j], a[i, j]))


def test_shift_of_every_action_at_one_node():
    """The MCTS call: one readout repeated to (|A|, d), a = arange(|A|)."""
    cfg = cfg_of()
    net = net_of(cfg)
    R = net.readout(*history(1, 3, 6))[0, -1]
    got = net.shift(R.expand(A, -1), torch.arange(A))
    want = torch.stack([oracle.shift(net, cfg, R, torch.tensor(k)) for k in range(A)])
    assert_close(got, want)
    assert not torch.allclose(got[0], got[1])


CLIPS = [  # L = 4 over the tiny A: n_{s+1} = 1 or not, the last states
    make_clip([["jump"], ["attack"], ["use", "use"]], [0.0] * 4, [0] * 4, cid="a"),
    make_clip([["forward", "turn_up"], ["hotbar_3"], ["left_click_1"]], [0.0] * 4, [0] * 4, cid="b"),
    make_clip([["use"], ["jump", "attack", "jump"], ["turn_up"]], [0.0] * 4, [0] * 4, cid="c"),
]  # fmt: skip


def test_forward_against_composition():
    """(7): logits (m, L, |A|) and the shift of a->_s (m, L, D), C by (1), a-> by (7)."""
    cfg = cfg_of()
    net = net_of(cfg)
    Z = torch.randn(3, 4, D, generator=torch.Generator().manual_seed(7), dtype=F64)
    lam, sh = net(Z, CLIPS)
    assert lam.shape == (3, 4, A) and sh.shape == (3, 4, D)
    want = oracle.forward(net, cfg, NAMES, Z, CLIPS)
    assert_close(lam, want[0])
    assert_close(sh, want[1])


def test_forward_clips_independent():
    """Row i depends only on Z_i and clip i."""
    net = net_of()
    Z = torch.randn(3, 4, D, generator=torch.Generator().manual_seed(8), dtype=F64)
    lam, sh = net(Z, CLIPS)
    lam1, sh1 = net(Z[[1]], CLIPS[1:2])
    assert_close(lam1[0], lam[1])
    assert_close(sh1[0], sh[1])


def test_float32():
    """Parameters and inputs float32: outputs float32, close to the float64 oracle."""
    cfg = cfg_of()
    net = Policy(cfg, D, NAMES).eval()
    randomize_gains(net)
    Z = torch.randn(3, 4, D, generator=torch.Generator().manual_seed(9))
    lam, sh = net(Z, CLIPS)
    assert lam.dtype == sh.dtype == torch.float32
    want = oracle.forward(net.double(), cfg, NAMES, Z.double(), CLIPS)
    assert_close(lam.double(), want[0], rtol=1e-4, atol=1e-5)
    assert_close(sh.double(), want[1], rtol=1e-4, atol=1e-5)
