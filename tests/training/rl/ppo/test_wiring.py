"""The tiny last-readout Policy wiring of specs/ppo.md Interface, in train mode."""

import math

import torch
from models.policy.policy_oracle import cfg_of

from mcpref.models.policy.network import Policy
from mcpref.training.rl.ppo.loss import ppo_loss, shift_loss
from mcpref.training.rl.ppo.metra import metra_loss, unit_skill

F64 = torch.float64


def test_last_readout_feeds_all_losses_with_action_dropout_active():
    names = ("a", "b", "c")
    policy = Policy(cfg_of(layers=1, action_dropout=0.5), 2, names).double().train()
    generator = torch.Generator().manual_seed(31)
    Y = torch.randn(2, 3, 2, generator=generator, dtype=F64, requires_grad=True)
    C = torch.rand(2, 3, 4, generator=generator, dtype=F64)
    C = C / C.sum(dim=-1, keepdim=True)
    actions = torch.tensor([0, 2], dtype=torch.int64)
    actual = 0.01 * torch.randn(2, 2, generator=generator, dtype=F64)
    actual.requires_grad_()
    alpha = torch.tensor([1.5, 0.25], dtype=F64)

    torch.manual_seed(19)
    torch.rand(2, 3, dtype=torch.float32)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    readouts = policy.readout(Y, C)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    R = readouts[:, -1]
    assert R.shape == (2, 8)
    without_dropout = policy.eval().readout(Y.detach(), C)[:, -1]
    assert not torch.allclose(R.detach(), without_dropout)
    policy.train()

    ppo = ppo_loss(
        policy.logits(R),
        actions,
        torch.tensor([0.4, 0.3], dtype=F64),
        torch.tensor([1.0, -0.5], dtype=F64),
        0.2,
    )
    predicted = policy.shift(R, actions)
    shift = shift_loss(
        predicted,
        actual,
        lambda delta: torch.linalg.vector_norm(alpha * delta, ord=2, dim=-1),
    )
    skill = unit_skill(predicted, alpha, 1.0e-8)
    log_lambda = torch.tensor(math.log(30.0), dtype=F64, requires_grad=True)
    metra = metra_loss(actual, skill, alpha, log_lambda, 1.0e-3)

    ppo_Y, ppo_head = torch.autograd.grad(
        ppo.sum(), (Y, policy.head.weight), retain_graph=True
    )
    shift_Y, shift_head = torch.autograd.grad(
        shift.sum(), (Y, policy.dynamics.w2.weight)
    )
    metra_actual, metra_dual, metra_shift = torch.autograd.grad(
        metra.sum(),
        (actual, log_lambda, policy.dynamics.w2.weight),
        allow_unused=True,
    )
    assert ppo.shape == shift.shape == metra.shape == (2,)
    assert not skill.requires_grad and skill.grad_fn is None
    assert ppo_Y.abs().sum() > 0 and ppo_head.grad_fn is None
    assert ppo_head.abs().sum() > 0
    assert shift_Y.abs().sum() > 0 and shift_head.abs().sum() > 0
    assert metra_actual.abs().sum() > 0 and metra_dual.abs().sum() > 0
    assert metra_shift is None
