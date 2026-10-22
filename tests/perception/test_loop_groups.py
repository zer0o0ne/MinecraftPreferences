"""The protocol of specs/pretrain.md with this block's modules (specs/perception.md Tests):
check (R) 4 of the pretraining loop with `Phi`, `PhiLoss` and the starting `groups` of
configs/pretrain.yaml ($\\pi$, its loss and $P$ faked with the groups of specs/policy.md and
specs/preference.md, tests/pretrain/loop_starting.py). The run is stopped by C's missing
`clips` marker at (R) 5, so reaching it shows that (R) 4 passed."""

from pretrain.index_world import drop, edit
from pretrain.loop_run import Loop, aborted
from pretrain.loop_starting import starting
from progress_bars import screen

from mcpref.perception.loss import PhiLoss
from mcpref.perception.phi import Phi
from mcpref.pretrain.config import load_config

from .world import backbone, config, tokenizer

DECLARED = ["phi_vision", "phi_vision_no_decay", "phi_llm", "phi_llm_no_decay",
            "phi_head", "phi_head_no_decay", "phi_dual", "pi", "pi_no_decay", "P",
            "P_no_decay"]  # fmt: skip


def modules(tmp_path) -> dict:
    cfg = config(tmp_path / "perception")
    return starting(phi=Phi(backbone(), tokenizer(), cfg), phi_loss=PhiLoss(cfg))


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
    six, phi_loss's phi_dual, pi's two, P's two)."""
    groups = {k: v for k, v in starting_groups(repo).items() if k != "phi_llm"}
    loop = Loop(tmp_path / "loop", groups=groups).write()
    o = loop.run(modules(tmp_path), raises=RuntimeError)
    aborted(o, f"the modules declare the groups {DECLARED}, the config {list(groups)}")
