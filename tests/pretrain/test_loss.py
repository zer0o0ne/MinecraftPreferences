"""`step_gradients` of specs/pretrain.md (Interface; (1), (2), (7)) against the float64
oracle of (1) on the whole mini-batch (loop_oracle.terms): mini-batches with and without
pairs, a clip in two items, $b_\\mu$ dividing $b$, not dividing it, $\\ge b$; the gradients
accumulated into `.grad`; the calls of a micro-batch in order, $P$ never with $m = 0$; each
term's gradient reaching $\\phi$; without $P$."""

import dataclasses

import pytest
import torch
from torch.testing import assert_close

import mcpref.pretrain.loss as loss_module
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.loss import gradient_steps, step_gradients

from .loop_configs import CONFIG
from .loop_modules import PhiLoss, PiLoss, PLoss, fakes, shared_log
from .loop_oracle import terms
from .loop_world import clip, p, u

M, K_TOT, R_TOT, W_PHI, W_PI, W_P = 5, 11, 3, 0.4, 0.7, 1.9
MIXED = [u(0), p(0), u(1), p(1)]  # q1 in p(0) and p(1)
REPEATS = [p(0), u(2), u(2), p(2)]  # u2 in two items; q0 in two pairs
UNLABELED = [u(0), u(3), u(0)]
CFG = PretrainConfig(**(CONFIG | {"betas": tuple(CONFIG["betas"])}))


def reader(log: list | None = None):
    """read(p): the Clip at p (loop_world.clip), recorded in `log`."""

    def read(path: str):
        if log is not None:
            log.append(("read", None, path))
        return clip(path)

    return read


def oracle(items: list, mods: dict) -> dict:
    """The gradients of (1) on the whole mini-batch: parameter name -> grad (None kept)."""
    loss, *_ = terms(items, mods, reader(), M, K_TOT, R_TOT, W_PHI, W_PI, W_P)
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
        items, mods, read or reader(), M, K_TOT, r_tot, W_PHI, W_PI, W_P, micro, cfg=CFG
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


def test_zero_phi_weight_keeps_other_gradient_paths():
    mods = fakes(noise=False, stateful=False)
    got = step_gradients(
        MIXED, mods, reader(), M, K_TOT, R_TOT, 0.0, W_PI, W_P, 2, cfg=CFG
    )
    mods_want = fakes(noise=False, stateful=False)
    want = terms(MIXED, mods_want, reader(), M, K_TOT, R_TOT, 0.0, W_PI, W_P)[0]
    want.backward()
    assert got["loss"] == pytest.approx(want.item(), rel=1e-12)
    assert_grads(grads(mods), grads(mods_want))


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
    got = step_gradients(
        UNLABELED, mods, reader(), M, K_TOT, 0, W_PHI, W_PI, W_P, 2, cfg=CFG
    )
    want = oracle(UNLABELED, fakes(pairs=False, noise=False, stateful=False))
    assert_grads(grads(mods), want)
    assert got["P"] is None


def test_cfg_is_required_keyword_only():
    with pytest.raises(TypeError):
        step_gradients(
            UNLABELED, fakes(), reader(), M, K_TOT, R_TOT, W_PHI, W_PI, W_P, 2
        )


@pytest.mark.parametrize("micro", [1, 3, 4])
def test_logged_losses(micro):
    """(7) without lr: the means of l_phi, l_pi over K, of l_P over R, and L(B), floats."""
    mods = fakes(noise=False, stateful=False)
    got = call(MIXED, mods, micro)
    loss, l_phi, l_pi, l_P = terms(MIXED, fakes(noise=False, stateful=False), reader(),
                                   M, K_TOT, R_TOT, W_PHI, W_PI, W_P)  # fmt: skip
    assert {
        "phi",
        "pi",
        "P",
        "loss",
        "counts",
        "components",
        "observations",
    } <= got.keys()
    assert all(type(got[k]) is float for k in ("phi", "pi", "P", "loss"))
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
    """Reads stay in path order; module calls stay in micro-batch order while only the
    immediate next micro-batch may be read during the current one."""
    items = [u(0), p(0), u(1), u(2), p(1)]
    mods = fakes(noise=False, stateful=False)
    log = shared_log(mods)
    call(items, mods, 2, reader(log))
    runs = [[u(0), p(0)], [u(1), u(2)], [p(1)]]
    expected = [[x for it in run for x in ([it["clip"]] if "clip" in it else [it["a"], it["b"]])] for run in runs]  # fmt: skip
    assert [c[2] for c in log if c[0] == "read"] == [x for run in expected for x in run]
    calls = [c for c in log if c[0] != "read"]
    for paths, run in zip(expected, runs):
        names = ["phi", "phi_loss", "pi", "pi_loss"]
        pairs = [it for it in run if "clip" not in it]
        names += ["P", "P_loss"] if pairs else []
        current, calls = calls[: len(names)], calls[len(names) :]
        assert [c[0] for c in current] == names
        ids = [x.rsplit("/", 1)[1] for x in paths]
        assert current[0][2]["ids"] == ids
        assert current[2][2]["ids"] == ids and current[2][2]["Z"].equal(
            current[1][2]["Z"]
        )
        if pairs:
            Z = current[1][2]["Z"]
            ia = [paths.index(it["a"]) for it in pairs]
            ib = [i + 1 for i in ia]
            assert current[4][2]["Za"].equal(Z[ia]) and current[4][2]["Zb"].equal(Z[ib])
            assert current[5][2]["labels"] == [it["labels"] for it in pairs]
    assert calls == []


def test_prefetch_matches_sequential_losses_gradients_reads_and_rng():
    def sequential(read, batches):
        for paths in batches:
            yield [read(path) for path in paths]

    def run():
        mods, reads = fakes(noise=True, stateful=False), []
        torch.manual_seed(812)
        logged = call(MIXED, mods, 2, reader(reads))
        return logged, grads(mods), reads, torch.get_rng_state()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("mcpref.pretrain.loss.prefetch_clips", sequential)
        want = run()
    got = run()
    for key in ("phi", "pi", "P", "loss", "dual_lambda"):
        assert got[0][key] == want[0][key]
    assert_grads(got[1], want[1])
    assert got[2] == want[2]
    assert got[3].equal(want[3])


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


def test_detached_observations_counts_and_bounded_contract():
    got = call(MIXED, fakes(noise=False, stateful=False), 2)
    assert got["counts"] == {"items": 4, "clips": 6, "pairs": 2, "comparable": 2,
                             "incomparable": 0, "sum_p": 1.0, "action_targets": 6,
                             "shift_targets": 6, "phi_changed": 12, "phi_state_pairs": 12}  # fmt: skip
    observations = got["observations"]
    assert [f.name for f in dataclasses.fields(observations)] == [
        "preference",
        "policy",
        "phi",
        "raw_policy",
    ]
    preference = observations.preference
    assert (
        preference.s.shape
        == preference.t.shape
        == preference.p.shape
        == preference.inc.shape
        == (2,)
    )
    assert (
        preference.s.device
        == preference.t.device
        == preference.p.device
        == preference.inc.device
        == torch.device("cpu")
    )
    assert (
        preference.s.dtype == preference.t.dtype == preference.p.dtype == torch.float32
    )
    assert preference.inc.dtype == torch.uint8
    tensors = [preference.s, preference.t, preference.p, preference.inc, observations.policy.q_sum,
               observations.policy.calibration_count, observations.policy.calibration_confidence,
               observations.policy.calibration_correct]  # fmt: skip
    assert all(not tensor.requires_grad for tensor in tensors)
    assert (
        observations.policy.q_sum.dtype == torch.float64
        and observations.policy.q_sum.sum() == 1
    )
    assert observations.policy.action_valid == observations.policy.shift_valid == 6
    assert observations.phi.z_count == 18 and observations.phi.state_pairs == 12
    assert not hasattr(observations.policy, "logits") and not hasattr(
        observations.phi, "Z"
    )


def override_policy_diagnostics(mods, q, current):
    terms = mods["pi_loss"].terms

    def replaced(*args):
        losses, valid, observations = terms(*args)
        return losses, valid, observations | {"q": q, "current": current}

    mods["pi_loss"].terms = replaced


def test_action_target_count_is_valid_rows_not_nonzero_elements():
    mods = fakes(noise=False, stateful=False)
    q = torch.tensor([[[0.5, 0.5, 0.0]]], dtype=torch.float64).repeat(6, 1, 1)
    override_policy_diagnostics(mods, q, q.clone())
    got = call(MIXED, mods, 6)
    assert (q > 0).sum() == 12 and got["counts"]["action_targets"] == 6
    torch.testing.assert_close(
        got["observations"].policy.q_sum,
        torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64),
    )


def test_copy_eps_zero_with_unreachable_target_is_null_not_infinite():
    mods = fakes(noise=False, stateful=False)
    q = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    current = torch.zeros_like(q)
    override_policy_diagnostics(mods, q, current)
    cfg = dataclasses.replace(CFG, action_copy_eps=0.0)
    got = step_gradients(
        [u(0)], mods, reader(), M, K_TOT, R_TOT, W_PHI, W_PI, W_P, 1, cfg=cfg
    )
    assert got["observations"].policy.copy_current is None
    assert got["observations"].policy.uniform == pytest.approx(
        torch.log(torch.tensor(3.0)).item()
    )


def test_no_pairs_has_null_preference_observations_and_counts():
    got = call(UNLABELED, fakes(noise=False, stateful=False), 2)
    assert got["observations"].preference is None
    assert got["counts"]["pairs"] == got["counts"]["comparable"] == 0
    assert got["counts"]["sum_p"] is None and got["components"]["P"] is None


def test_pair_only_additive_succ_uses_all_pairs_not_comparable_only():
    single = call([p(0)], fakes(noise=False, stateful=False), 1)
    mixed = call([p(0), p(2)], fakes(noise=False, stateful=False), 2)
    assert mixed["counts"]["pairs"] == 2 and mixed["counts"]["comparable"] == 1
    assert mixed["counts"]["incomparable"] == 1 and mixed["counts"]["sum_p"] == 1.0
    assert mixed["components"]["P"]["succ"] == pytest.approx(
        single["components"]["P"]["succ"] / 2
    )
    assert mixed["P"] == pytest.approx(
        mixed["components"]["P"]["inc"] + mixed["components"]["P"]["succ"]
    )


def test_action_top1_tie_uses_smallest_action_index():
    mods = fakes(noise=False, stateful=False)
    with torch.no_grad():
        mods["pi"].Wa.zero_()
        mods["pi"].ba.zero_()
    q = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
    override_policy_diagnostics(mods, q, q.clone())
    got = call([u(0)], mods, 1)
    assert got["observations"].policy.correct_sum == 0
    assert got["observations"].policy.confidence_sum == pytest.approx(1 / 3)
    q = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    override_policy_diagnostics(mods, q, q.clone())
    got = call([u(0)], mods, 1)
    assert got["observations"].policy.correct_sum == 1


def test_host_transfers_are_deferred_until_after_last_backward(monkeypatch):
    mods = fakes(noise=False, stateful=False)
    events, backwards = [], []
    original_backward = torch.Tensor.backward
    original_item = torch.Tensor.item
    original_cpu = torch.Tensor.cpu

    def backward(self, *args, **kwargs):
        result = original_backward(self, *args, **kwargs)
        backwards.append(None)
        return result

    def item(self):
        events.append(("item", len(backwards)))
        return original_item(self)

    def cpu(self, *args, **kwargs):
        events.append(("cpu", len(backwards)))
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    call(MIXED, mods, 1)
    assert events and all(count == len(MIXED) for _, count in events)


def test_tensor_compilation_is_limited_to_ordinary_steps(monkeypatch, tmp_path):
    calls = []
    real_micro = loss_module.micro_loss

    def record_micro(*args, compile_tensors=False):
        calls.append(compile_tensors)
        return real_micro(*args, compile_tensors=compile_tensors)

    monkeypatch.setattr(loss_module, "micro_loss", record_micro)
    monkeypatch.setattr(loss_module, "collect_micro", lambda *args: {"micro": args[-1]})
    args = (MIXED, fakes(noise=False, stateful=False), reader())
    constants = (M, K_TOT, R_TOT, W_PHI, W_PI, W_P, 2, CFG)
    gradient_steps(*args, *constants[:-1], cfg=CFG, selected=None)
    assert calls == [True, True]

    calls.clear()
    args = (MIXED, fakes(noise=False, stateful=False), reader())
    gradient_steps(*args, *constants[:-1], cfg=CFG, selected=(None, tmp_path, None))
    assert calls == [False, False]
