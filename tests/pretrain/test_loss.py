"""`step_gradients` of specs/pretrain.md (Interface; (1), (2), (7)) against the float64
oracle of (1) on the whole mini-batch (loop_oracle.terms): mini-batches with and without
pairs, a clip in two items, $b_\\mu$ dividing $b$, not dividing it, $\\ge b$; the gradients
accumulated into `.grad`; the calls of a micro-batch in order, $P$ never with $m = 0$; each
term's gradient reaching $\\phi$; without $P$."""

import pytest
import torch
from torch.testing import assert_close

from mcpref.pretrain.loss import step_gradients

from .loop_modules import PhiLoss, PiLoss, PLoss, fakes, shared_log
from .loop_oracle import terms
from .loop_world import clip, p, u

M, K_TOT, R_TOT, W_PI, W_P = 5, 11, 3, 0.7, 1.9
MIXED = [u(0), p(0), u(1), p(1)]  # q1 in p(0) and p(1)
REPEATS = [p(0), u(2), u(2), p(2)]  # u2 in two items; q0 in two pairs
UNLABELED = [u(0), u(3), u(0)]


def reader(log: list | None = None):
    """read(p): the Clip at p (loop_world.clip), recorded in `log`."""

    def read(path: str):
        if log is not None:
            log.append(("read", None, path))
        return clip(path)

    return read


def oracle(items: list, mods: dict) -> dict:
    """The gradients of (1) on the whole mini-batch: parameter name -> grad (None kept)."""
    loss, *_ = terms(items, mods, reader(), M, K_TOT, R_TOT, W_PI, W_P)
    loss.backward()
    return {
        f"{k}.{n}": q.grad for k, m in mods.items() for n, q in m.named_parameters()
    }


def grads(mods: dict) -> dict:
    return {
        f"{k}.{n}": q.grad for k, m in mods.items() for n, q in m.named_parameters()
    }


def call(items: list, mods: dict, micro: int, read=None, r_tot: int = R_TOT) -> dict:
    return step_gradients(
        items, mods, read or reader(), M, K_TOT, r_tot, W_PI, W_P, micro
    )


def assert_grads(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for k, g in want.items():
        assert (got[k] is None) == (g is None), k
        if g is not None:
            assert_close(got[k], g, msg=k)


@pytest.mark.parametrize("micro", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("items", [MIXED, REPEATS], ids=["mixed", "repeats"])
def test_gradients_of_eq_1(items, micro):
    """b = 4: b_mu = 1, 2 (dividing), 3 (not), 4, 6 (>= b): the accumulated gradients equal
    those of (1) on the whole mini-batch."""
    mods = fakes(noise=False, stateful=False)
    call(items, mods, micro)
    assert_grads(grads(mods), oracle(items, fakes(noise=False, stateful=False)))


@pytest.mark.parametrize("micro", [1, 2, 5])
def test_mini_batch_without_pairs(micro):
    """No pair: P and P_loss never called, P's gradients stay None, `P` None."""
    mods = fakes(noise=False, stateful=False)
    got = call(UNLABELED, mods, micro)
    assert mods["P"].calls == mods["P_loss"].calls == 0
    assert all(q.grad is None for q in mods["P"].parameters())
    assert got["P"] is None
    assert_grads(grads(mods), oracle(UNLABELED, fakes(noise=False, stateful=False)))


def test_without_P_modules():
    """r_tot = 0 (no pair in the training part): the four modules suffice."""
    mods = fakes(pairs=False, noise=False, stateful=False)
    got = step_gradients(UNLABELED, mods, reader(), M, K_TOT, 0, W_PI, W_P, 2)
    want = oracle(UNLABELED, fakes(pairs=False, noise=False, stateful=False))
    assert_grads(grads(mods), want)
    assert got["P"] is None


@pytest.mark.parametrize("micro", [1, 3, 4])
def test_logged_losses(micro):
    """(7) without lr: the means of l_phi, l_pi over K, of l_P over R, and L(B), floats."""
    mods = fakes(noise=False, stateful=False)
    got = call(MIXED, mods, micro)
    loss, l_phi, l_pi, l_P = terms(MIXED, fakes(noise=False, stateful=False), reader(),
                                   M, K_TOT, R_TOT, W_PI, W_P)  # fmt: skip
    assert set(got) == {"phi", "pi", "P", "loss"}
    assert all(type(v) is float for v in got.values())
    assert got["phi"] == pytest.approx(l_phi.mean().item(), rel=1e-12)
    assert got["pi"] == pytest.approx(l_pi.mean().item(), rel=1e-12)
    assert got["P"] == pytest.approx(l_P.mean().item(), rel=1e-12)
    assert got["loss"] == pytest.approx(loss.item(), rel=1e-12)


def test_accumulates_into_existing_gradients():
    """`.grad` not zeroed: an existing gradient plus the gradient of (1)."""
    mods = fakes(noise=False, stateful=False)
    start = {}
    for k, m in mods.items():
        for n, q in m.named_parameters():
            q.grad = torch.full_like(q, 0.25)
            start[f"{k}.{n}"] = q.grad.clone()
    call(MIXED, mods, 3)
    want = oracle(MIXED, fakes(noise=False, stateful=False))
    for k, g in grads(mods).items():
        assert_close(g, start[k] + want[k], msg=k)


def test_calls_of_each_micro_batch_in_order():
    """b_mu = 2 on [u0, p0, u1, u2, p1]: per micro-batch its clips read, then phi on K in
    item order (a, then b), l_phi, pi on Z and the clips K, l_pi, and with a pair P on the
    rows i_a, i_b of Z and l_P on the labels; nothing read ahead."""
    items = [u(0), p(0), u(1), u(2), p(1)]
    mods = fakes(noise=False, stateful=False)
    log = shared_log(mods)
    call(items, mods, 2, reader(log))
    runs = [[u(0), p(0)], [u(1), u(2)], [p(1)]]
    for run in runs:
        paths = [x for it in run for x in ([it["clip"]] if "clip" in it else [it["a"], it["b"]])]  # fmt: skip
        reads = []
        while log[0][0] == "read":
            reads.append(log.pop(0)[2])
        assert set(reads) == set(paths)
        names = ["phi", "phi_loss", "pi", "pi_loss"]
        pairs = [it for it in run if "clip" not in it]
        names += ["P", "P_loss"] if pairs else []
        calls, log[:] = log[: len(names)], log[len(names) :]
        assert [c[0] for c in calls] == names
        ids = [x.rsplit("/", 1)[1] for x in paths]
        assert calls[0][2]["ids"] == ids
        assert calls[2][2]["ids"] == ids and calls[2][2]["Z"].equal(calls[1][2]["Z"])
        if pairs:
            Z = calls[1][2]["Z"]
            ia = [paths.index(it["a"]) for it in pairs]
            ib = [i + 1 for i in ia]
            assert calls[4][2]["Za"].equal(Z[ia]) and calls[4][2]["Zb"].equal(Z[ib])
            assert calls[5][2]["labels"] == [it["labels"] for it in pairs]
    assert log == []


@pytest.mark.parametrize("live", ["phi_loss", "pi_loss", "P_loss"])
def test_each_term_reaches_phi(live):
    """No stop-gradient added: with the other two losses cut from their inputs (their own
    stop-gradients), phi's gradient is that of the live term alone, not zero."""

    def build() -> dict:
        mods = fakes(noise=False, stateful=False)
        mods["phi_loss"] = PhiLoss(stateful=False, cut=live != "phi_loss")
        mods["pi_loss"] = PiLoss(cut=live != "pi_loss")
        mods["P_loss"] = PLoss(cut=live != "P_loss")
        return mods

    mods = build()
    call(MIXED, mods, 2)
    want = oracle(MIXED, build())
    assert_grads(grads(mods), want)
    assert want["phi.A"].abs().sum() > 0
