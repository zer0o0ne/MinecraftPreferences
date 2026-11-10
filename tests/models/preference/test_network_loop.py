"""`Preference` and `PreferenceLoss` inside `pretrain` of specs/pretrain.md (the protocol;
specs/preference.md Interface) with fake $\\phi$ and $\\pi$, test-owned starting groups, labels
of specs/acyclic.md (4), against the loop oracle of tests/pretrain run with $P$ and $\\ell^P$
transcribed (preference_oracle.py) on the same initial weights."""

import torch
from torch import nn
from training.pretrain.loop_checks import assert_run
from training.pretrain.loop_modules import D
from training.pretrain.loop_oracle import train
from training.pretrain.loop_run import Loop
from training.pretrain.loop_starting import starting
from training.pretrain.loop_world import LABELED

from mcpref.models.preference.loss import PreferenceLoss
from mcpref.models.preference.network import Preference
from models.preference.preference_oracle import cfg_of, decayed_names, loss, preference

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


def test_preference_in_the_loop(tmp_path):
    """Every step's losses, checkpoint and model.pt against the oracle; P trained."""
    from training.pretrain.loop_starting import starting_groups

    loop = Loop(
        tmp_path, world=LABELED, keep_checkpoints=20, groups=starting_groups()
    ).write()
    loop.run(components(Preference(CFG, D).double(), PreferenceLoss()))
    make = lambda: components(RefP(), RefLoss())
    steps, states = train(loop.world, loop.config, make)
    assert_run(loop, steps, states, kept=20, make=make)
    assert {s.losses["P"] is None for s in steps} == {True, False}
    start = Preference(CFG, D).double().state_dict()
    trained = loop.model()["P"]
    assert list(trained) == list(start)
    assert all(not torch.equal(trained[k], start[k]) for k in start)
