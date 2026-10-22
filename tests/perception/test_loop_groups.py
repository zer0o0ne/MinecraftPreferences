"""The protocol of specs/pretrain.md with this block's modules (specs/perception.md Tests):
check (R) 4 of the pretraining loop with `Phi`, `PhiLoss` and the starting `groups` of
configs/pretrain.yaml (pi and its loss without parameters; P, block 5's, faked with its
groups `P` and `P_no_decay`). The run is stopped by C's missing `clips` marker at (R) 5, so
reaching it shows that (R) 4 passed."""

from pretrain.index_world import drop, edit
from pretrain.loop_modules import fakes
from pretrain.loop_run import Loop, aborted
from progress_bars import screen
from torch import nn

from mcpref.perception.loss import PhiLoss
from mcpref.perception.phi import Phi
from mcpref.pretrain.config import load_config

from .world import backbone, config, tokenizer

DECLARED = ["phi_vision", "phi_vision_no_decay", "phi_llm", "phi_llm_no_decay",
            "phi_head", "phi_head_no_decay", "phi_dual", "P", "P_no_decay"]  # fmt: skip


class NoParameters(nn.Module):
    """pi or its loss before block 6: no parameters, no groups."""

    def groups(self) -> dict:
        return {}


def modules(tmp_path) -> dict:
    mods = fakes()
    P = mods["P"]
    P.groups = lambda: {"P": [P.u, P.v], "P_no_decay": [P.beta]}
    cfg = config(tmp_path / "perception")
    return {"phi": Phi(backbone(), tokenizer(), cfg), "phi_loss": PhiLoss(cfg),
            "pi": NoParameters(), "pi_loss": NoParameters(), "P": P,
            "P_loss": mods["P_loss"]}  # fmt: skip


def starting_groups(repo) -> dict:
    return dict(load_config(repo / "configs/pretrain.yaml").groups)


def test_starting_groups_pass(tmp_path, repo):
    groups = starting_groups(repo)
    assert list(groups) == DECLARED
    loop = Loop(tmp_path / "loop", groups=groups).write()
    edit(loop.repo / "w/clips/run.json", drop("clips"))
    o = loop.run(modules(tmp_path), raises=RuntimeError)
    aborted(o, "w/clips is not complete")
    assert screen(o.err) == []


def test_declared_groups_in_order(tmp_path, repo):
    """A config group missing: the message lists the declared groups in group order (phi's
    six, phi_loss's phi_dual, P's two)."""
    groups = {k: v for k, v in starting_groups(repo).items() if k != "phi_llm"}
    loop = Loop(tmp_path / "loop", groups=groups).write()
    o = loop.run(modules(tmp_path), raises=RuntimeError)
    aborted(o, f"the modules declare the groups {DECLARED}, the config {list(groups)}")
