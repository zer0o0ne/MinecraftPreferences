"""`Preference` and `PreferenceLoss` inside `pretrain` of specs/pretrain.md (the protocol;
specs/preference.md Interface) with fake $\\phi$ and $\\pi$ (tests/pretrain/loop_starting.py)
declaring the starting `groups` of configs/pretrain.yaml, labels of specs/acyclic.md (4),
against the loop oracle of tests/pretrain run with $P$ and $\\ell^P$ transcribed
(preference_oracle.py) on the same initial weights."""

import torch
from pretrain.loop_checks import assert_run
from pretrain.loop_modules import D
from pretrain.loop_oracle import train
from pretrain.loop_run import Loop
from pretrain.loop_starting import starting
from pretrain.loop_world import LABELED
from torch import nn

from mcpref.preferences.loss import PreferenceLoss
from mcpref.preferences.network import Preference
from mcpref.pretrain.config import load_config

from .preference_oracle import cfg_of, decayed_names, loss, preference

CFG = cfg_of(history=2, width=4, heads=2, hidden=6)  # L_P = 2 < L = 3, d_h = 2


class RefP(nn.Module):
    """The submodules of a fresh Preference (same names, same initial weights); forward
    and groups transcribed."""

    def __init__(self):
        super().__init__()
        net = Preference(CFG, D).double()
        self.embed, self.layers, self.norm = net.embed, net.layers, net.norm
        self.pref, self.inc = net.pref, net.inc

    def forward(self, Za: torch.Tensor, Zb: torch.Tensor) -> tuple:
        return preference(self, CFG, Za, Zb)

    def groups(self) -> dict:
        named = dict(self.named_parameters())
        decay, rest = decayed_names(CFG)
        return {"P": [named[n] for n in decay], "P_no_decay": [named[n] for n in rest]}


class RefLoss(nn.Module):
    def forward(self, out: tuple, labels: list) -> torch.Tensor:
        return loss(out, labels)

    def groups(self) -> dict:
        return {}


def components(P: nn.Module, P_loss: nn.Module) -> dict:
    return starting(P=P, P_loss=P_loss)


def test_preference_in_the_loop(tmp_path, repo):
    """Every step's losses, checkpoint and model.pt against the oracle; P trained."""
    groups = load_config(repo / "configs/pretrain.yaml").groups
    loop = Loop(tmp_path, world=LABELED, keep_checkpoints=20, groups=groups).write()
    loop.run(components(Preference(CFG, D).double(), PreferenceLoss()))
    make = lambda: components(RefP(), RefLoss())
    steps, states = train(loop.world, loop.config, make)
    assert_run(loop, steps, states, kept=20, make=make)
    assert {s.losses["P"] is None for s in steps} == {True, False}
    start = Preference(CFG, D).double().state_dict()
    trained = loop.model()["P"]
    assert list(trained) == list(start)
    assert all(not torch.equal(trained[k], start[k]) for k in start)
