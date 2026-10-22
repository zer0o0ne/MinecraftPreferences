"""(10), (11), (12) of specs/perception.md for `PhiLoss`: the changed pairs on hand cases
(meta text, the pixel threshold strictly, integer differences, the cursor); the loss of each
clip against a float64 oracle; the gradient structure (log lambda only from its own term,
sg(lambda) and sg(q) in phi's, chi_s = 0 pairs, d_s = 0 with zero gradient); log lambda's
start and group."""

import math

import numpy as np
import torch

from mcpref.perception.loss import PhiLoss

from . import oracle
from .world import F64, H, W, clip, config, line, values

MOUSE = {"x": 240.0, "y": 120.0}


def luma(y: int) -> np.ndarray:
    return np.full((H, W), y)


def meta_clip():
    """chi = (0, 1, 0, 1) by hand (tau = 1): yaw 0 -> 360 and pitch 0 -> 0.01 keep the meta
    text ('0.0'); the hotbar changes; RGB 1 -> 0 everywhere: mean |diff| = 1 = tau, not
    above (and no uint8 wrap-around); RGB 0 -> 2: mean 2 > tau."""
    lumas = [luma(17), luma(17), luma(17), luma(16), luma(18)]  # RGB 1, 1, 1, 0, 2
    lines = [line(), line(yaw=360.0, pitch=0.01), line(hotbar=3), line(hotbar=3),
             line(hotbar=3)]  # fmt: skip
    return clip(lumas, lines, [0, 1, 2, 3, 4], [False] * 5, [1, 1, 3, 10])


def cursor_clip():
    """chi = (1, 0, 1, 0) by hand: equal metas and frames, the cursor drawn on states 1 and
    2 (g true, same mouse) only: it appears, stays, disappears, stays away."""
    lines = [line(mouse=MOUSE) for _ in range(5)]
    gui = [False, True, True, False, False]
    return clip([luma(17)] * 5, lines, [0, 1, 2, 3, 4], gui, [2, 1, 10, 1])


CHI = [[0, 1, 0, 1], [1, 0, 1, 0]]


def module(tmp_path, dtype=F64, **changes) -> PhiLoss:
    return PhiLoss(config(tmp_path, **changes)).to(dtype)


def log_lambda(loss: PhiLoss) -> torch.nn.Parameter:
    (p,) = loss.groups()["phi_dual"]
    return p


def latent() -> torch.Tensor:
    """Z of the two clips: pair 1 of clip 0 (chi 1) has z_2 = z_1 (d = 0)."""
    z = np.random.default_rng(0).normal(scale=0.8, size=(2, 5, 4))
    z[0, 2] = z[0, 1]
    return torch.tensor(z, requires_grad=True)


def test_oracle_chi_by_hand(tmp_path):
    """The oracle of (10) agrees with the hand values (a check of the oracle itself)."""
    v = values(tmp_path)
    assert [oracle.chi(meta_clip(), v), oracle.chi(cursor_clip(), v)] == CHI


def test_loss_against_the_oracle(tmp_path):
    """(11) of each clip, shape (n,), in Z's dtype; chi from clip_states by (10)."""
    loss = module(tmp_path)
    Z = latent()
    got = loss(Z, [meta_clip(), cursor_clip()])
    assert got.shape == (2,) and got.dtype == F64
    w = np.ones(2)
    want, _, _ = oracle.loss(Z.detach().numpy(), CHI, [[1, 1, 3, 10], [2, 1, 10, 1]], w,
                             values(tmp_path), math.log(30.0))  # fmt: skip
    torch.testing.assert_close(got, torch.tensor(want))


def coverage(Z: np.ndarray, ns: list, v: dict) -> None:
    """The case covers both branches of the clamp and of each norm term."""
    alpha = np.asarray(v["alpha"])
    d2 = [float(np.sum((alpha * (z[s + 1] - z[s])) ** 2)) for z in Z for s in range(4)]
    n2 = [n**2 for row in ns for n in row]
    active = [b - a < v["dual_slack"] for a, b in zip(d2, n2)]
    assert any(active) and not all(active)
    norms = (Z**2).sum(-1).ravel()
    assert (norms > v["c"] ** 2).any() and (norms < v["c"] ** 2).any()


def test_gradients_against_the_oracle(tmp_path):
    """sum_i w_i l_i: dZ with lambda and q held fixed in their stop-gradient places, d_s = 0
    giving zero gradient (no NaN); d log lambda = sum_i w_i mean_s chi_s q_s only."""
    loss = module(tmp_path)
    Z = latent()
    ns = [[1, 1, 3, 10], [2, 1, 10, 1]]
    w = np.array([0.7, -1.3])
    v = values(tmp_path)
    coverage(Z.detach().numpy(), ns, v)
    (loss(Z, [meta_clip(), cursor_clip()]) * torch.tensor(w)).sum().backward()
    _, dZ, dlog = oracle.loss(Z.detach().numpy(), CHI, ns, w, v, math.log(30.0))
    assert torch.isfinite(Z.grad).all()
    torch.testing.assert_close(Z.grad, torch.tensor(dZ))
    torch.testing.assert_close(log_lambda(loss).grad, torch.tensor(dlog, dtype=F64))


def test_zero_distance_zero_gradient(tmp_path):
    """A clip whose states all share one z, every pair changed: d_s = 0, q_s = eps; the
    gradient on Z is that of the norm terms only, finite."""
    loss = module(tmp_path)
    z = torch.tensor(np.tile([[0.9, -0.8, 0.7, 0.6]], (5, 1))[None], requires_grad=True)
    c = meta_clip()
    c.info["n"][:] = [1, 2, 3, 4]
    loss(z, [c]).sum().backward()
    norm_only = 0.7 * 2 * z.detach() * np.array([1, 2, 2, 2, 1])[None, :, None] / 4
    torch.testing.assert_close(z.grad, norm_only)
    assert float(z.detach()[0, 0] @ z.detach()[0, 0]) > 1.5**2


def test_unchanged_pairs_leave_the_distance_and_constraint(tmp_path):
    """tau large, equal metas: chi = 0 everywhere: only the norm terms; no gradient on
    log lambda."""
    loss = module(tmp_path, tau=1.0e9)
    c = cursor_clip()
    Z = latent()[1:].detach().requires_grad_()
    got = loss(Z, [c])
    v = values(tmp_path, tau=1.0e9)
    want, dZ, dlog = oracle.loss(Z.detach().numpy(), [[0] * 4], [c.info["n"]], np.ones(1), v,
                                 math.log(30.0))  # fmt: skip
    assert dlog == 0
    torch.testing.assert_close(got, torch.tensor(want))
    got.sum().backward()
    torch.testing.assert_close(Z.grad, torch.tensor(dZ))
    g = log_lambda(loss).grad
    assert g is None or g == 0


def test_tau_zero_any_change(tmp_path):
    """tau = 0: a pixel change of 1/(3 H W) on average is a change."""
    lumas = [luma(17), luma(17)]
    lumas[1] = lumas[1].copy()
    lumas[1][0, 0] = 16
    c = clip(lumas, [line(), line()], [0, 1], [False, False], [1])
    v = values(tmp_path, tau=0)
    assert oracle.chi(c, v) == [1]
    Z = torch.tensor(np.random.default_rng(2).normal(size=(1, 2, 4)))
    want, _, _ = oracle.loss(Z.numpy(), [[1]], [[1]], np.ones(1), v, math.log(30.0))
    torch.testing.assert_close(module(tmp_path, tau=0)(Z, [c]), torch.tensor(want))


def test_log_lambda_start_and_group(tmp_path):
    """log lambda starts at ln(dual_init); the only parameter, in the group phi_dual; the
    loss uses lambda = exp(log lambda) of the current value."""
    loss = PhiLoss(config(tmp_path, dual_init=2.5))
    p = log_lambda(loss)
    assert list(loss.groups()) == ["phi_dual"]
    assert [id(q) for q in loss.parameters()] == [id(p)]
    assert abs(float(p.detach()) - math.log(2.5)) < 1e-6
    loss.to(F64)
    with torch.no_grad():
        log_lambda(loss).fill_(0.3)
    Z = latent().detach()
    want, _, _ = oracle.loss(Z.numpy(), CHI, [[1, 1, 3, 10], [2, 1, 10, 1]], np.ones(2),
                             values(tmp_path, dual_init=2.5), 0.3)  # fmt: skip
    torch.testing.assert_close(
        loss(Z, [meta_clip(), cursor_clip()]), torch.tensor(want)
    )


def test_float32(tmp_path):
    """Computed in Z's dtype: float32 Z and parameter give a float32 loss."""
    loss = module(tmp_path, torch.float32)
    Z = latent().detach()
    got = loss(Z.float(), [meta_clip(), cursor_clip()])
    assert got.dtype == torch.float32
    want, _, _ = oracle.loss(Z.numpy(), CHI, [[1, 1, 3, 10], [2, 1, 10, 1]], np.ones(2),
                             values(tmp_path), math.log(30.0))  # fmt: skip
    torch.testing.assert_close(got, torch.tensor(want, dtype=torch.float32))
