"""`PreferenceLoss` of src/mcpref/preferences/loss.py against specs/preference.md (10): the
four terms of specs/acyclic.md (5) with both orders from two calls of $P$, hand values, no
term of $s$ (and no NaN) for `inc` = 1, the dtype, no clamp, the gradient reaching $Z$."""

import math

import pytest
import torch
from torch.testing import assert_close

from mcpref.preferences.loss import PreferenceLoss
from mcpref.preferences.network import Preference

from .preference_oracle import cfg_of, randomize_gains

F64, D = torch.float64, 5
LABELS = [{"p": 0.7, "inc": 0}, {"p": None, "inc": 1}, {"p": 0.0, "inc": 0},
          {"p": 1.0, "inc": 0}, {"p": None, "inc": 1}]  # fmt: skip


def bce(x: float, y: float) -> float:
    """BCE of Notation in Python floats, log sigma stable on both sides."""

    def log_sigmoid(z: float) -> float:
        return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))

    return -(y * log_sigmoid(x) + (1 - y) * log_sigmoid(-x))


def contract(ab: tuple, ba: tuple, labels: list) -> list[float]:
    """specs/acyclic.md (5) on the logits of both orders."""
    out = []
    for r, y in enumerate(labels):
        value = bce(float(ab[1][r]), y["inc"]) + bce(float(ba[1][r]), y["inc"])
        if y["inc"] == 0:
            value += bce(float(ab[0][r]), y["p"]) + bce(float(ba[0][r]), 1 - y["p"])
        out.append(value)
    return out


def test_against_both_orders():
    """(10) on P's logits of one call against the four terms from two calls."""
    net = Preference(cfg_of(), D).double()
    randomize_gains(net)
    g = torch.Generator().manual_seed(1)
    Ya, Yb = torch.randn(5, 3, D, generator=g, dtype=F64), torch.randn(5, 2, D, generator=g, dtype=F64)  # fmt: skip
    with torch.no_grad():
        ab, ba = net(Ya, Yb), net(Yb, Ya)
    got = PreferenceLoss()(ab, LABELS)
    assert got.shape == (5,) and got.dtype == F64
    assert_close(got, torch.tensor(contract(ab, ba, LABELS), dtype=F64))


def test_hand_values():
    """s = t = 0: each BCE is log 2; inc = 0 -> 4 log 2, inc = 1 -> 2 log 2. s = 2, t = -1,
    p = 0.25: 2 BCE(-1, 0) + 2 BCE(2, 0.25) = 2 log(1 + e^-1) + 2 (0.75 * 2 + log(1 + e^-2))."""
    s, t = torch.tensor([0.0, 0.0, 2.0], dtype=F64), torch.tensor([0.0, 0.0, -1.0], dtype=F64)  # fmt: skip
    labels = [{"p": 0.5, "inc": 0}, {"p": None, "inc": 1}, {"p": 0.25, "inc": 0}]
    third = 2 * math.log1p(math.exp(-1)) + 2 * (1.5 + math.log1p(math.exp(-2)))
    want = torch.tensor([4 * math.log(2), 2 * math.log(2), third], dtype=F64)
    assert_close(PreferenceLoss()((s, t), labels), want)


def test_one_pair():
    s, t = torch.tensor([1.0], dtype=F64), torch.tensor([0.5], dtype=F64)
    got = PreferenceLoss()((s, t), [{"p": 0.3, "inc": 0}])
    assert got.shape == (1,)
    assert_close(got, torch.tensor([2 * bce(0.5, 0) + 2 * bce(1.0, 0.3)], dtype=F64))


def test_incomparable_has_no_term_of_s():
    """inc = 1: the loss does not depend on s, its gradient on s is 0 (not NaN), even for
    large |s|; inc = 0 pairs keep theirs."""
    labels = [{"p": None, "inc": 1}, {"p": 0.4, "inc": 0}, {"p": None, "inc": 1}]
    t = torch.tensor([0.3, -0.2, 1.5], dtype=F64)
    for s0 in ([500.0, 1.0, -500.0], [0.0, 1.0, 3.0]):
        s = torch.tensor(s0, dtype=F64, requires_grad=True)
        got = PreferenceLoss()((s, t), labels)
        want = [2 * bce(0.3, 1), 2 * bce(-0.2, 0) + 2 * bce(1.0, 0.4), 2 * bce(1.5, 1)]
        assert_close(got.detach(), torch.tensor(want, dtype=F64))
        got.sum().backward()
        assert torch.isfinite(s.grad).all()
        assert s.grad[0] == 0 and s.grad[2] == 0 and s.grad[1] != 0


def test_no_clamp():
    """Large logits: the exact BCE (e.g. BCE(-50, 1) = 50 + log(1 + e^-50)), no clamping."""
    s, t = (
        torch.tensor([50.0, -40.0], dtype=F64),
        torch.tensor([-50.0, 60.0], dtype=F64),
    )
    labels = [{"p": 0.0, "inc": 0}, {"p": 1.0, "inc": 0}]
    want = [2 * bce(-50, 0) + 2 * bce(50, 0), 2 * bce(60, 0) + 2 * bce(-40, 1)]
    assert_close(PreferenceLoss()((s, t), labels), torch.tensor(want, dtype=F64))
    assert want[0] > 99  # a clamp of sigma at 1e-7 would give 2 * 16.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_of_the_logits(dtype):
    s, t = torch.tensor([0.4, -1.2], dtype=dtype), torch.tensor([2.0, 0.1], dtype=dtype)
    labels = [{"p": 0.9, "inc": 0}, {"p": None, "inc": 1}]
    got = PreferenceLoss()((s, t), labels)
    assert got.dtype == dtype
    want = [2 * bce(2.0, 0) + 2 * bce(0.4, 0.9), 2 * bce(0.1, 1)]
    tol = {"rtol": 1e-6, "atol": 1e-6} if dtype == torch.float32 else {}
    assert_close(got, torch.tensor(want, dtype=dtype), **tol)


def test_no_groups():
    loss = PreferenceLoss()
    assert loss.groups() == {} and list(loss.parameters()) == []


def test_gradient_reaches_z_and_p():
    """The loss's gradient reaches the used rows of Z_a and Z_b (none of the older ones)
    and every parameter of P; with only inc = 1 pairs w_succ gets a zero gradient."""
    net = Preference(cfg_of(), D).double()
    g = torch.Generator().manual_seed(2)
    Za = torch.randn(3, 5, D, generator=g, dtype=F64, requires_grad=True)
    Zb = torch.randn(3, 3, D, generator=g, dtype=F64, requires_grad=True)
    labels = [{"p": 0.7, "inc": 0}, {"p": None, "inc": 1}, {"p": 0.2, "inc": 0}]
    PreferenceLoss()(net(Za, Zb), labels).sum().backward()
    assert (Za.grad[:, :2] == 0).all() and (Za.grad[:, 2:].abs().sum((0, 2)) > 0).all()
    assert (Zb.grad.abs().sum((0, 2)) > 0).all()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    net.zero_grad()
    PreferenceLoss()(net(Za, Zb), [{"p": None, "inc": 1}] * 3).sum().backward()
    assert (net.pref.weight.grad == 0).all() and net.inc.weight.grad.abs().sum() > 0
