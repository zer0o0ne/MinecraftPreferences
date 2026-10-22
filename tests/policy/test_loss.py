"""`PolicyLoss` of src/mcpref/policy/loss.py against specs/policy.md (9): a float64 oracle
(policy_oracle.py) on the hand clip and random clips with the approved masks; both terms and
$w_\\Delta$; the rows the loss never reads (the placeholder shifts, the last two states); the
metric as given, on $\\delta$ of shape $(m, L - 2, D)$; the target's gradient stopped; the
gradient reaching $Z$ through $\\pi$'s input, $W_c$ and $E$ receiving gradients; the dtype;
no parameters, no groups."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close
from transformer.layers_oracle import randomize_gains

from mcpref.policy.loss import PolicyLoss
from mcpref.policy.network import Policy

from . import policy_oracle as oracle
from .policy_oracle import cfg_of
from .policy_world import HAND, make_clip, random_clips

F64, D = torch.float64, 3
ALPHA = torch.tensor([1.0, 2.0, 0.5], dtype=F64)
W = 1.5  # shift_weight of cfg_of()


def clips_of(L: int, seed: int) -> list:
    """The hand clip (L = 8) or random clips of L states."""
    return [HAND] if L == 8 else random_clips(seed, 4, L)


def outputs(m: int, L: int, A: int, seed: int, dtype=F64) -> tuple:
    g = torch.Generator().manual_seed(seed)
    lam = 3 * torch.randn(m, L, A, generator=g, dtype=dtype)
    return lam, torch.randn(m, L, D, generator=g, dtype=dtype), torch.randn(m, L, D, generator=g, dtype=dtype)  # fmt: skip


def module(masks, metric=None, **changes) -> PolicyLoss:
    return PolicyLoss(cfg_of(**changes), masks, metric or oracle.alpha_metric(ALPHA))


@pytest.mark.parametrize("L, seed", [(8, 0), (3, 1), (5, 2), (6, 3)], ids=["hand", "L 3", "L 5", "L 6"])  # fmt: skip
def test_against_oracle(masks, L, seed):
    """(9) of each clip: shape (m,), the oracle's value."""
    clips = clips_of(L, seed)
    lam, sh, Z = outputs(len(clips), L, len(masks.names), seed)
    got = module(masks)((lam, sh), Z, clips)
    want = oracle.loss((lam, sh), Z, clips, masks.names, W, oracle.alpha_metric(ALPHA))
    assert got.shape == (len(clips),) and got.dtype == F64
    assert_close(got, want)


def test_hand_clip_terms(masks):
    """The hand clip: CE over the 6 states with a next interval (states 3, 5 with q = 0 add
    nothing), plus w times the squared d_alpha of state 0's shift error (chi only there),
    over L - 2 = 6."""
    lam, sh, Z = outputs(1, 8, len(masks.names), 4)
    q, _ = oracle.targets(HAND, masks.names)
    logp = torch.log_softmax(lam[0], -1)
    ce = -sum(float((torch.tensor(q[s]) * logp[s]).sum()) for s in range(6))
    err = sh[0, 0] - (Z[0, 1] - Z[0, 0])
    want = (ce + W * float(((ALPHA * err) ** 2).sum())) / 6
    assert module(masks)((lam, sh), Z, [HAND]).item() == pytest.approx(want, rel=1e-12)


def test_shift_weight(masks):
    """w_Delta multiplies the shift term alone."""
    clips = clips_of(5, 5)
    lam, sh, Z = outputs(4, 5, len(masks.names), 5)
    l1, l2, l0 = (module(masks, shift_weight=w)((lam, sh), Z, clips) for w in (1.0, 3.0, 1e-300))  # fmt: skip
    assert_close(l2 - l0, 3 * (l1 - l0))
    assert (l1 - l0 > 0).any()


def test_rows_never_read(masks):
    """The shifts of the states with chi = 0 (the placeholders included, also huge) and the
    logits of states L - 2, L - 1 change nothing; their gradients are 0."""
    lam, sh, Z = outputs(1, 8, len(masks.names), 6)
    lam.requires_grad_()
    sh.requires_grad_()
    loss = module(masks)((lam, sh), Z, [HAND])
    loss.sum().backward()
    lam2, sh2 = lam.detach().clone(), sh.detach().clone()
    sh2[0, 1:] = 1e200
    lam2[0, 6:] = 50.0
    assert_close(module(masks)((lam2, sh2), Z, [HAND]), loss.detach())
    assert not sh.grad[0, 1:].any() and not lam.grad[0, 6:].any()
    assert sh.grad[0, 0].abs().sum() > 0 and lam.grad[0, :6].abs().sum() > 0
    q, _ = oracle.targets(HAND, masks.names)
    assert not lam.grad[0, 3].any() and not lam.grad[0, 5].any()  # q = 0: no CE term
    assert not q[3].any() and not q[5].any()


def test_metric_as_given(masks):
    """The metric given to the loss (here a weighted L1 norm) is the one used, called on
    delta of shape (m, L - 2, D)."""
    shapes = []

    def metric(delta: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(delta.shape))
        return (torch.tensor([0.3, 1.0, 2.0], dtype=delta.dtype) * delta.abs()).sum(-1)

    clips = clips_of(6, 7)
    lam, sh, Z = outputs(4, 6, len(masks.names), 7)
    got = module(masks, metric)((lam, sh), Z, clips)
    assert shapes and set(shapes) == {(4, 4, D)}
    shapes.clear()
    assert_close(got, oracle.loss((lam, sh), Z, clips, masks.names, W, metric))


def test_target_gradient_stopped(masks):
    """With outputs not depending on Z, no gradient reaches Z."""
    clips = clips_of(5, 8)
    lam, sh, Z = outputs(4, 5, len(masks.names), 8)
    sh.requires_grad_()
    Z.requires_grad_()
    module(masks)((lam, sh), Z, clips).sum().backward()
    assert Z.grad is None or not Z.grad.any()
    assert sh.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(masks, dtype):
    """The loss in the dtype of the logits; float32 close to float64."""
    clips = clips_of(6, 9)
    lam, sh, Z = outputs(4, 6, len(masks.names), 9, dtype)
    got = module(masks)((lam, sh), Z, clips)
    assert got.dtype == dtype
    want = oracle.loss((lam.double(), sh.double()), Z.double(), clips, masks.names, W,
                       oracle.alpha_metric(ALPHA))  # fmt: skip
    assert_close(got.double(), want, rtol=1e-5, atol=1e-5)


def test_no_state(masks):
    loss = module(masks)
    assert (
        list(loss.parameters()) == []
        and loss.state_dict() == {}
        and loss.groups() == {}
    )


def test_every_target_masked_gives_zero(masks):
    """q = 0 and chi = 0 at every state: the loss is 0."""
    clip = make_clip([["jump"], ["close"], ["hotbar_1"], ["close"]], [0.0] * 5, [0] * 5)
    lam, sh, Z = outputs(1, 5, len(masks.names), 10)
    assert module(masks)((lam, sh), Z, [clip]).item() == 0.0


def test_gradients_through_the_policy(masks):
    """pi(Z, clips) (eval mode) into the loss: Z's gradient that of the oracle (through pi's
    input only; rows L - 2, L - 1 none: causal, targets stopped); W_c's gradient nonzero;
    E's gradient nonzero exactly on the rows of a->_s with chi_s."""
    cfg = cfg_of(history=8)  # L_pi = L
    clips = [HAND]
    net = Policy(cfg, D, masks.names).double().eval()
    randomize_gains(net)
    Z = torch.randn(1, 8, D, generator=torch.Generator().manual_seed(11), dtype=F64)
    Zg = Z.clone().requires_grad_()
    module(masks)(net(Zg, clips), Zg, clips).sum().backward()
    Zo = Z.clone().requires_grad_()
    out = oracle.forward(net, cfg, masks.names, Zo, clips)
    want = oracle.loss(out, Zo, clips, masks.names, W, oracle.alpha_metric(ALPHA))
    want.sum().backward()
    assert_close(Zg.grad, Zo.grad)
    assert not Zg.grad[0, 6:].any() and Zg.grad[0, :6].abs().sum() > 0
    assert net.executed.weight.grad.abs().sum() > 0
    rows = (net.actions.weight.grad != 0).any(1).nonzero().flatten().tolist()
    _, chi = oracle.targets(HAND, masks.names)
    a = oracle.next_actions(HAND, masks.names)
    assert rows == sorted({a[s] for s in np.flatnonzero(chi)})


def test_shift_in_the_graph_without_chi(masks):
    """No state with chi (every target masked): the shift term removed by `torch.where`,
    so E and the shift MLP get zero gradients, not None (AdamW then decays them,
    specs/pretrain.md (6))."""
    clip = make_clip([["jump"], ["close"], ["hotbar_1"], ["close"]], [0.0] * 5, [0] * 5)
    net = Policy(cfg_of(history=5), D, masks.names).double().eval()
    Z = torch.randn(1, 5, D, generator=torch.Generator().manual_seed(12), dtype=F64)
    module(masks)(net(Z, [clip]), Z, [clip]).sum().backward()
    for p in [net.actions.weight, *net.dynamics.parameters()]:
        assert p.grad is not None and not p.grad.any()
