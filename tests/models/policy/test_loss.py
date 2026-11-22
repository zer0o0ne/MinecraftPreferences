"""`PolicyLoss` of src/mcpref/models/policy/loss.py against specs/policy.md (9): a float64 oracle
(policy_oracle.py) on the hand clip and random clips with the approved masks; both terms and
$w_\\Delta$; the rows the loss never reads (the placeholder shifts, the last state); the
metric as given, on $\\delta$ of shape $(m, L - 1, D)$; the target's gradient stopped; the
gradient reaching $Z$ through $\\pi$'s input, $W_c$ and $E$ receiving gradients; the dtype;
no parameters, no groups."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from mcpref.models.policy.loss import PolicyLoss
from mcpref.models.policy.network import Policy
from models.policy import policy_oracle as oracle
from models.policy.policy_oracle import cfg_of
from models.policy.policy_world import HAND, make_clip, random_clips
from models.transformer.layers_oracle import randomize_gains

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
    """The hand clip: CE over seven transitions (states 4, 6 masked), with valid shifts
    at states 0, 1, 3, averaged over all seven transitions."""
    lam, sh, Z = outputs(1, 8, len(masks.names), 4)
    q, _ = oracle.targets(HAND, masks.names)
    logp = torch.log_softmax(lam[0], -1)
    ce = -sum(float((torch.tensor(q[s]) * logp[s]).sum()) for s in range(7))
    err = sh[0, [0, 1, 3]] - (Z[0, [1, 2, 4]] - Z[0, [0, 1, 3]])
    want = (ce + W * float(((ALPHA * err) ** 2).sum())) / 7
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
    logits of terminal state L - 1 change nothing; their gradients are 0."""
    lam, sh, Z = outputs(1, 8, len(masks.names), 6)
    lam.requires_grad_()
    sh.requires_grad_()
    loss = module(masks)((lam, sh), Z, [HAND])
    loss.sum().backward()
    lam2, sh2 = lam.detach().clone(), sh.detach().clone()
    sh2[0, [2, 4, 5, 6, 7]] = 1e200
    lam2[0, 7:] = 50.0
    assert_close(module(masks)((lam2, sh2), Z, [HAND]), loss.detach())
    assert not sh.grad[0, [2, 4, 5, 6, 7]].any() and not lam.grad[0, 7:].any()
    assert sh.grad[0, 0].abs().sum() > 0 and lam.grad[0, :7].abs().sum() > 0
    q, _ = oracle.targets(HAND, masks.names)
    assert not lam.grad[0, 4].any() and not lam.grad[0, 6].any()  # q = 0: no CE term
    assert not q[4].any() and not q[6].any()


def test_metric_as_given(masks):
    """The metric given to the loss (here a weighted L1 norm) is the one used, called on
    delta of shape (m, L - 1, D)."""
    shapes = []

    def metric(delta: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(delta.shape))
        return (torch.tensor([0.3, 1.0, 2.0], dtype=delta.dtype) * delta.abs()).sum(-1)

    clips = clips_of(6, 7)
    lam, sh, Z = outputs(4, 6, len(masks.names), 7)
    got = module(masks, metric)((lam, sh), Z, clips)
    assert shapes and set(shapes) == {(4, 5, D)}
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
    clip = make_clip(
        [["close"], ["close"], ["hotbar_1"], ["close"]], [0.0] * 5, [0] * 5
    )
    lam, sh, Z = outputs(1, 5, len(masks.names), 10)
    assert module(masks)((lam, sh), Z, [clip]).item() == 0.0


def test_gradients_through_the_policy(masks):
    """pi(Z, clips) (eval mode) into the loss: Z's gradient that of the oracle (through pi's
    input only; terminal row L - 1 none: causal, targets stopped); W_c's gradient nonzero;
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
    assert not Zg.grad[0, 7:].any() and Zg.grad[0, :7].abs().sum() > 0
    assert net.executed.weight.grad.abs().sum() > 0
    rows = (net.actions.weight.grad != 0).any(1).nonzero().flatten().tolist()
    _, chi = oracle.targets(HAND, masks.names)
    a = oracle.next_actions(HAND, masks.names)
    assert rows == sorted({a[s] for s in np.flatnonzero(chi)})


def test_shift_in_the_graph_without_chi(masks):
    """No state with chi (every target masked): the shift term removed by `torch.where`,
    so E and the shift MLP get zero gradients, not None (AdamW then decays them,
    specs/pretrain.md (6))."""
    clip = make_clip(
        [["close"], ["close"], ["hotbar_1"], ["close"]], [0.0] * 5, [0] * 5
    )
    net = Policy(cfg_of(history=5), D, masks.names).double().eval()
    Z = torch.randn(1, 5, D, generator=torch.Generator().manual_seed(12), dtype=F64)
    module(masks)(net(Z, [clip]), Z, [clip]).sum().backward()
    for p in [net.actions.weight, *net.dynamics.parameters()]:
        assert p.grad is not None and not p.grad.any()


def test_terms_components_validity_and_forward(masks):
    """Eq. (3): state action/shift terms, valid-row masks and the original per-clip total."""
    clips = [HAND]
    lam, sh, Z = outputs(1, 8, len(masks.names), 21)
    loss = module(masks)
    terms, valid, observations = loss.terms((lam, sh), Z, clips)
    q, chi = oracle.targets(HAND, masks.names)
    logp = torch.log_softmax(lam[0], -1)
    action = -sum(float((torch.tensor(q[s]) * logp[s]).sum()) for s in range(7)) / 7
    err = sh[0, [0, 1, 3]] - (Z[0, [1, 2, 4]] - Z[0, [0, 1, 3]])
    shift = W * float(((ALPHA * err) ** 2).sum()) / 7
    assert set(terms) == {"action", "shift", "change", "total"}
    assert set(observations) == {"q", "current", "zero_shift"}
    assert terms["action"].item() == pytest.approx(action, rel=1e-12)
    assert terms["shift"].item() == pytest.approx(shift, rel=1e-12)
    assert_close(terms["total"], terms["action"] + terms["shift"])
    assert_close(loss((lam, sh), Z, clips), terms["total"])
    assert valid["action_valid"].dtype == valid["shift_valid"].dtype == torch.bool
    assert valid["action_valid"].shape == valid["shift_valid"].shape == (1, 7)
    assert valid["action_valid"].tolist() == [(torch.tensor(q).sum(-1) > 0).tolist()]
    assert valid["shift_valid"].tolist() == [chi.tolist()]


def test_terms_computes_targets_once(masks, monkeypatch):
    import mcpref.models.policy.loss as loss_module

    real = loss_module.targets
    calls = []

    def counting(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(loss_module, "targets", counting)
    loss = module(masks)
    lam, sh, Z = outputs(1, 8, len(masks.names), 22)
    loss.terms((lam, sh), Z, [HAND])
    assert len(calls) == 1


def test_weights_shared_and_observations_are_per_call(masks, monkeypatch):
    import mcpref.models.policy.loss as loss_module
    import mcpref.models.policy.network as network_module

    real = network_module.weights
    calls = []

    def counting(*args):
        calls.append(args[0])
        return real(*args)

    monkeypatch.setattr(network_module, "weights", counting)
    monkeypatch.setattr(loss_module, "weights", counting, raising=False)
    net = Policy(cfg_of(history=8, action_dropout=0.0), D, masks.names).double().eval()
    loss = module(masks)
    clips = [HAND, random_clips(23, 1, 8)[0]]
    Z = torch.randn(2, 8, D, generator=torch.Generator().manual_seed(24), dtype=F64)
    returned = []
    for index, clip in enumerate(clips):
        out = net(Z[index : index + 1], [clip])
        _, valid, observations = loss.terms(out, Z[index : index + 1], [clip])
        q, chi = oracle.targets(clip, masks.names)
        current = oracle.input_weights(clip, masks.names)[:-1]
        assert observations["current"].data_ptr() == out[2].data_ptr()
        assert_close(observations["q"], torch.tensor(q[None], dtype=F64))
        assert_close(observations["current"], torch.tensor(current[None], dtype=F64))
        factual = Z[index : index + 1, 1:] - Z[index : index + 1, :-1]
        zero = W * oracle.alpha_metric(ALPHA)(factual).square()
        zero = torch.where(torch.tensor(chi), zero, 0.0).sum(dim=1) / 7
        assert_close(observations["zero_shift"], zero)
        assert valid["shift_valid"].tolist() == [chi.tolist()]
        returned.append(observations)
    assert calls == clips
    q0, _ = oracle.targets(HAND, masks.names)
    assert_close(returned[0]["q"], torch.tensor(q0[None], dtype=F64))
    assert not returned[0]["q"].equal(returned[1]["q"])


def test_change_uses_prior_membership_current_multiplicity_and_known_idle(masks):
    clip = make_clip(
        [
            ["forward", "attack"],
            ["forward", "forward", "jump", "close"],
            [],
            ["jump", "jump"],
            ["jump", "close"],
        ],
        [0.0] * 6,
        [0] * 6,
    )
    count = len(masks.names)
    logits = torch.zeros((1, 6, count), dtype=F64, requires_grad=True)
    probability = torch.tensor(
        [[0.1, 0.8, 0.2, 0.3, 0.4, 0.9]], dtype=F64, requires_grad=True
    )
    current = torch.tensor(oracle.input_weights(clip, masks.names)[None], dtype=F64)
    Z, shift = torch.zeros((1, 6, D), dtype=F64), torch.zeros((1, 6, D), dtype=F64)
    terms, _, _ = module(masks).terms((logits, shift, current, probability), Z, [clip])
    # First row unknown; close masked; repeated A occurs twice; idle is known.
    log_count = np.log(count)
    assert terms["change"].item() == pytest.approx((0.6 + 0.7 + 0.4) * log_count / 5)
    terms["change"].sum().backward()
    torch.testing.assert_close(
        probability.grad,
        torch.tensor([[0, 1 / 3, 0, -1, 1, 0]], dtype=F64) * log_count / 5,
    )
    assert not logits.grad[:, [0, 2, 5]].any()
    assert logits.grad[:, [1, 3, 4]].abs().sum() > 0


def test_prior_frequency_does_not_scale_change_repeat_label(masks):
    clips = [
        make_clip([past, ["forward", "jump"]], [0.0] * 3, [0] * 3)
        for past in (["forward", "attack"], ["forward"] * 5 + ["attack"])
    ]
    lam, sh, Z = outputs(1, 3, len(masks.names), 51)
    p = torch.full((1, 3), 0.8, dtype=F64)
    values = []
    for clip in clips:
        C = torch.tensor(oracle.input_weights(clip, masks.names)[None], dtype=F64)
        values.append(module(masks).terms((lam, sh, C, p), Z, [clip])[0]["change"])
    assert_close(values[0], values[1])
