"""Runs of `main` of specs/policy.md (13) on the worlds of loop_world.py: a `Loop` whose module
configs are valid (perception with $D$ = 2 and its cursor sprite, the approved action-mask
catalogue linked under the temporary repository root, a policy with $L_\\pi = L$, a preference
with $L_P \\le L$) and whose groups are the starting ones; fake builders of $\\phi$ (no
download) and, where a test says so, of $\\phi$'s loss and of the loop."""

import contextlib
import dataclasses
import io
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from perception.world import write_config as write_perception
from policy.policy_oracle import cfg_of as policy_cfg
from preferences.preference_oracle import cfg_of as preference_cfg
from progress_bars import tqdm_defaults

import mcpref.perception.loss
import mcpref.perception.phi
import mcpref.pretrain.loop
import mcpref.pretrain.run
from mcpref.pretrain.run import main

from .index_run import repo_root
from .loop_configs import MODULE_CONFIGS, SCREENS
from .loop_modules import L
from .loop_run import Loop
from .loop_starting import PhiLossStart, PhiStart
from .loop_world import WORLD, World

PREFIX = "mcpref.pretrain.run: "
POLICY = policy_cfg(history=L, width=4, heads=2, hidden=6, shift_hidden=5)
PREFERENCE = preference_cfg(history=2, width=4, heads=2, hidden=6)  # L_P = 2 <= L = 3


class PhiFloat32(PhiStart):
    """The fake phi giving Z in float32, the dtype of the modules `main` builds."""

    def forward(self, clips: list) -> torch.Tensor:
        return super().forward(clips).float()


@dataclass
class Outcome:
    code: int | None
    err: str
    out: str


def dump(cfg) -> str:
    """A config dataclass as YAML (tuples as lists)."""
    values = dataclasses.asdict(cfg)
    values = {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}
    return yaml.safe_dump(values, sort_keys=False)


class Site(Loop):
    """A Loop with the starting groups of configs/pretrain.yaml and valid module configs."""

    def __init__(self, tmp: Path, repo: Path, world: World = WORLD, policy=POLICY,
                 preference=PREFERENCE, **changes):  # fmt: skip
        groups = yaml.safe_load((repo / "configs/pretrain.yaml").read_text())["groups"]
        super().__init__(tmp, world=world, **({"groups": groups} | changes))
        self.approved, self.policy, self.preference = repo, policy, preference
        self.built: list = []  # the configs given to the fake build_phi

    def write(self) -> "Site":
        super().write()
        at = {k: self.repo / v for k, v in MODULE_CONFIGS.items()}
        perception = write_perception(self.tmp / "perception", alpha=[1.0, 2.0])
        at["perception"].write_bytes(perception.read_bytes())
        at["policy"].write_text(dump(self.policy))
        at["preference"].write_text(dump(self.preference))
        (self.repo / SCREENS).unlink()
        os.symlink(self.approved / "configs/gui-screens.yaml", self.repo / SCREENS)
        return self

    def fakes(self, mp: pytest.MonkeyPatch, phi_loss: bool = False, loop=None) -> None:
        """build_phi faked (recording its config); `phi_loss`: PhiLoss faked too; `loop`:
        a replacement of `pretrain`. Patched in their modules and in run.py."""

        def build_phi(cfg):
            self.built.append(cfg)
            return PhiFloat32()

        patches = [(mcpref.perception.phi, "build_phi", build_phi)]
        if phi_loss:
            patches.append(
                (mcpref.perception.loss, "PhiLoss", lambda cfg: PhiLossStart())
            )
        if loop:
            patches.append((mcpref.pretrain.loop, "pretrain", loop))
        for module, name, value in patches:
            mp.setattr(module, name, value)
            mp.setattr(mcpref.pretrain.run, name, value, raising=False)

    def main(self, argv: list | None = None, phi_loss: bool = False, loop=None, patch=None) -> Outcome:  # fmt: skip
        """main on the config (or `argv`) under the repository root, tqdm with its defaults."""
        err, out = io.StringIO(), io.StringIO()
        with pytest.MonkeyPatch.context() as mp, repo_root(self.repo):
            tqdm_defaults(mp)
            mp.chdir(self.cwd)
            self.fakes(mp, phi_loss, loop)
            if patch:
                patch(mp)
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = main([str(self.path)] if argv is None else argv)
        return Outcome(code, err.getvalue(), out.getvalue())


def recorder(calls: list):
    """A `pretrain` recording its arguments."""
    return lambda cfg, path, modules: calls.append((cfg, path, modules))
