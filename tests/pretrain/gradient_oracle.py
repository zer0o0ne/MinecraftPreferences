"""Tiny independent gradient-flow fixtures and schedule helpers."""

import torch
from torch import nn

from mcpref.pretrain.checks import index_run
from mcpref.pretrain.config import load_config
from mcpref.pretrain.gradient_flow import select_gradient_steps
from mcpref.pretrain.items import part

from .index_run import repo_root
from .test_update import diagnostic_groups


class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.unused = nn.Parameter(torch.tensor([3.0]))


class PhiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dual = nn.Parameter(torch.tensor(0.5))


class Preference(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0]))


def modules() -> tuple[dict, object]:
    mods = {"phi": Phi(), "phi_loss": PhiLoss(), "P": Preference()}
    groups = diagnostic_groups(
        {
            "z": [mods["P"].w, mods["phi"].unused],
            "a": [mods["phi"].a, mods["phi_loss"].dual],
        },
        {
            mods["P"].w: "P.w",
            mods["phi"].unused: "phi.unused",
            mods["phi"].a: "phi.a",
            mods["phi_loss"].dual: "phi_loss.dual",
        },
    )
    return mods, groups


def weighted(mods: dict, factor=1.0) -> dict[str, torch.Tensor]:
    return {
        "phi": factor * (2 * mods["phi"].a.sum() + 3 * mods["phi_loss"].dual),
        "pi": factor * 5 * mods["phi"].a.sum(),
        "P": factor * (7 * mods["phi"].a.sum() + 11 * mods["P"].w.sum()),
    }


def select(loop) -> dict:
    with repo_root(loop.repo):
        cfg = load_config(loop.path)
        run = index_run(cfg)
        return select_gradient_steps(cfg, run, part(cfg, run))
