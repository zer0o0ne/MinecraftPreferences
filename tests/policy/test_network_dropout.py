"""The dropout of the executed actions, specs/policy.md (2): in training mode one draw per call
`torch.rand((m, n), dtype=float32) < p` from the global CPU generator over the used rows,
their c_j replaced by the one-hot of none; against a transcription of the draw
(policy_oracle.py); none in eval mode; p = 0."""

import pytest
import torch
from torch.testing import assert_close
from transformer.layers_oracle import randomize_gains

from mcpref.policy.network import Policy

from . import policy_oracle as oracle
from .policy_oracle import cfg_of
from .policy_world import NAMES, make_clip

F64, D, A = torch.float64, 5, len(NAMES)
HALF = cfg_of(action_dropout=0.5)  # L_pi = 4


def net_of(p: float) -> Policy:
    net = Policy(cfg_of(action_dropout=p), D, NAMES).double()
    randomize_gains(net)
    return net


def history(m: int, ell: int, seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    Y = torch.randn(m, ell, D, generator=g, dtype=F64)
    C = torch.rand(m, ell, A + 1, generator=g, dtype=F64) ** 3
    return Y, C / C.sum(-1, keepdim=True)


def state_after_draw(seed: int, m: int, n: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.rand((m, n), dtype=torch.float32)
    return torch.get_rng_state()


@pytest.mark.parametrize("ell", [3, 4, 7], ids=lambda x: f"l {x}")
def test_training_mode_against_transcription(ell):
    """p = 0.5 (rows dropped and kept), L_pi = 4: the draw over the n = min(4, l) used rows;
    the generator advanced by exactly that draw."""
    net = net_of(0.5).train()
    Y, C = history(3, ell, 1)
    n = min(4, ell)
    torch.manual_seed(5)
    b = oracle.draw(3, n, 0.5)
    assert b.any() and not b.all()
    torch.manual_seed(5)
    got = net.readout(Y, C)
    assert torch.equal(torch.get_rng_state(), state_after_draw(5, 3, n))
    assert_close(got, oracle.readout(net, HALF, Y, oracle.dropped(C, b)))
    assert not torch.allclose(got, oracle.readout(net, HALF, Y, C))


def test_one_draw_per_call():
    """Two calls in training mode: two draws (the second call's rows dropped by the next
    draw of the generator)."""
    net = net_of(0.5).train()
    Y, C = history(2, 4, 2)
    torch.manual_seed(6)
    b1, b2 = oracle.draw(2, 4, 0.5), oracle.draw(2, 4, 0.5)
    torch.manual_seed(6)
    first, second = net.readout(Y, C), net.readout(Y, C)
    assert_close(first, oracle.readout(net, HALF, Y, oracle.dropped(C, b1)))
    assert_close(second, oracle.readout(net, HALF, Y, oracle.dropped(C, b2)))


def test_eval_mode_draws_nothing():
    """Eval mode: no dropout, the global generator untouched."""
    net = net_of(0.5).eval()
    Y, C = history(2, 4, 3)
    torch.manual_seed(7)
    before = torch.get_rng_state()
    got = net.readout(Y, C)
    assert torch.equal(torch.get_rng_state(), before)
    assert_close(got, oracle.readout(net, HALF, Y, C))


def test_p_zero():
    """p = 0 in training mode: nothing dropped (the eval output); the draw still made."""
    net = net_of(0.0)
    Y, C = history(2, 4, 4)
    want = net.eval().readout(Y, C)
    torch.manual_seed(8)
    got = net.train().readout(Y, C)
    assert torch.equal(torch.get_rng_state(), state_after_draw(8, 2, 4))
    assert_close(got, want)


def test_p_close_to_one_drops_every_row():
    """p = 0.999: every executed action replaced by none (here all of the draw below p)."""
    net = net_of(0.999).train()
    Y, C = history(2, 4, 9)
    torch.manual_seed(9)
    assert oracle.draw(2, 4, 0.999).all()
    torch.manual_seed(9)
    got = net.readout(Y, C)
    none = torch.zeros(2, 4, A + 1, dtype=F64)
    none[..., -1] = 1.0
    assert_close(got, oracle.readout(net, cfg_of(action_dropout=0.999), Y, none))


CLIPS = [
    make_clip([["jump"], ["attack"], ["use", "use"]], [0.0] * 4, [0] * 4, cid="a"),
    make_clip([["forward", "turn_up"], ["hotbar_3"], ["left_click_1"]], [0.0] * 4, [0] * 4, cid="b"),
]  # fmt: skip


def test_forward_in_training_mode():
    """(7) in training mode: C by (1), then the draw of (2) over the L rows."""
    net = net_of(0.5).train()
    Z = torch.randn(2, 4, D, generator=torch.Generator().manual_seed(10), dtype=F64)
    torch.manual_seed(11)
    b = oracle.draw(2, 4, 0.5)
    torch.manual_seed(11)
    lam, sh = net(Z, CLIPS)
    want = oracle.forward(net, HALF, NAMES, Z, CLIPS, b)
    assert_close(lam, want[0])
    assert_close(sh, want[1])
