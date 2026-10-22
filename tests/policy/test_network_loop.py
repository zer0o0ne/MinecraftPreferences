"""`Policy` and `PolicyLoss` inside `pretrain` of specs/pretrain.md (the protocol: `pi(Z, clips)`,
`pi_loss(out, Z, clips)`; specs/policy.md Interface) with fake $\\phi$ and $P$
(tests/pretrain/loop_starting.py) declaring the starting `groups` of configs/pretrain.yaml,
against the loop oracle of tests/pretrain run with $\\pi$ (its dropout draw in training mode
included) and $\\ell^\\pi$ transcribed (policy_oracle.py) on the same initial weights; the
approved masks."""

import torch
from pretrain.loop_checks import assert_run
from pretrain.loop_modules import D, L
from pretrain.loop_oracle import train
from pretrain.loop_run import Loop
from pretrain.loop_starting import starting
from torch import nn

from mcpref.policy.loss import PolicyLoss
from mcpref.policy.network import Policy
from mcpref.pretrain.config import load_config

from . import policy_oracle as oracle
from .policy_oracle import cfg_of

CFG = cfg_of(history=L, width=4, heads=2, hidden=6, shift_hidden=5, action_dropout=0.5)
METRIC = oracle.alpha_metric(torch.tensor([0.5, 2.0], dtype=torch.float64))  # D = 2


class RefPi(nn.Module):
    """The submodules of a fresh Policy (same names, same initial weights); forward (with
    the draw of (2) in training mode) and groups transcribed."""

    def __init__(self, names: tuple):
        super().__init__()
        net = Policy(CFG, D, names).double()
        self.embed, self.executed, self.layers = net.embed, net.executed, net.layers
        self.norm, self.head, self.actions = net.norm, net.head, net.actions
        self.dynamics, self.names = net.dynamics, names

    def forward(self, Z: torch.Tensor, clips: list) -> tuple:
        b = oracle.draw(len(clips), L, CFG.action_dropout) if self.training else None
        return oracle.forward(self, CFG, self.names, Z, clips, b)

    def groups(self) -> dict:
        named = list(self.named_parameters())
        gain = lambda n: n.endswith("norm.weight")
        return {"pi": [p for n, p in named if not gain(n)],
                "pi_no_decay": [p for n, p in named if gain(n)]}  # fmt: skip


class RefPiLoss(nn.Module):
    def __init__(self, names: tuple):
        super().__init__()
        self.names = names

    def forward(self, out: tuple, Z: torch.Tensor, clips: list) -> torch.Tensor:
        return oracle.loss(out, Z, clips, self.names, CFG.shift_weight, METRIC)

    def groups(self) -> dict:
        return {}


def test_policy_in_the_loop(tmp_path, repo, masks):
    """Every step's losses, checkpoint and model.pt against the oracle; pi trained."""
    groups = load_config(repo / "configs/pretrain.yaml").groups
    loop = Loop(tmp_path, keep_checkpoints=20, groups=groups).write()
    pi = Policy(CFG, D, masks.names).double()
    loop.run(starting(pi=pi, pi_loss=PolicyLoss(CFG, masks, METRIC)))
    make = lambda: starting(pi=RefPi(masks.names), pi_loss=RefPiLoss(masks.names))
    steps, states = train(loop.world, loop.config, make)
    assert_run(loop, steps, states, kept=20, make=make)
    start = Policy(CFG, D, masks.names).double().state_dict()
    trained = loop.model()["pi"]
    assert list(trained) == list(start)
    assert all(not torch.equal(trained[k], start[k]) for k in start)
    assert loop.model()["pi_loss"] == {}
