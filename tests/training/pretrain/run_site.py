"""Runs of `main` of specs/policy.md (13) on the worlds of loop_world.py: a `Loop` whose module
configs are valid and test-owned (perception with $D$ = 2 and its cursor sprite, an action-mask
catalogue under the temporary repository root, a policy with $L_\\pi = L$, a preference with
$L_P \\le L$) and whose groups have the starting values; fake builders of $\\phi$ (no download)
and, where a test says so, of $\\phi$'s loss and of the loop."""

import contextlib
import dataclasses
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from models.perception.world import write_config as write_perception
from models.policy.policy_oracle import cfg_of as policy_cfg
from models.preference.preference_oracle import cfg_of as preference_cfg
from progress_bars import tqdm_defaults

import mcpref.models.factory
import mcpref.models.perception.loss
import mcpref.models.perception.phi
import mcpref.training.pretrain.loop
import mcpref.training.pretrain.run
from mcpref.training.pretrain.run import main
from training.pretrain.data.index_run import repo_root
from training.pretrain.loop_configs import MODULE_CONFIGS
from training.pretrain.loop_modules import L
from training.pretrain.loop_run import Loop
from training.pretrain.loop_starting import PhiLossStart, PhiStart
from training.pretrain.loop_world import WORLD, World

PREFIX = "mcpref.training.pretrain.run: "
POLICY = policy_cfg(history=L, width=4, heads=2, hidden=6, shift_hidden=5)
PREFERENCE = preference_cfg(history=2, width=4, heads=2, hidden=6)  # L_P = 2 <= L = 3


class PhiFloat32(PhiStart):
    """Seeded FP32 head; forward follows the dtype selected by the agent factory."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(2, 2, dtype=torch.float32))

    def forward(self, clips: list, states=None) -> torch.Tensor:
        x = (
            torch.tensor(np.stack([c.frames[:, :4] for c in clips]), dtype=self.A.dtype)
            / 255
        )
        z = x @ self.A.T + self.c
        if self.noise:
            z = z * (1 + 0.1 * torch.rand(len(clips), 1, 1, dtype=self.A.dtype))
        return z + self.e1 + self.e2 + self.e3 + self.e4

    def groups(self) -> dict:
        groups = super().groups()
        groups["phi_head"] = [*groups["phi_head"], self.head[-1].weight]
        groups["phi_head_no_decay"] = [
            *groups["phi_head_no_decay"],
            self.head[-1].bias,
        ]
        return groups


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
    """A Loop with test-owned starting groups and valid module configs."""

    def __init__(self, tmp: Path, repo: Path, world: World = WORLD, policy=POLICY,
                 preference=PREFERENCE, **changes):  # fmt: skip
        from training.pretrain.loop_starting import starting_groups

        super().__init__(tmp, world=world, **({"groups": starting_groups()} | changes))
        self.policy, self.preference = policy, preference
        self.built: list = []  # the configs given to the fake build_phi

    def write(self) -> "Site":
        from training.pretrain.loop_configs import write_action_mask_catalogue

        super().write()
        at = {k: self.repo / v for k, v in MODULE_CONFIGS.items()}
        perception = write_perception(self.tmp / "perception", alpha=[1.0, 2.0])
        at["perception"].write_bytes(perception.read_bytes())
        at["policy"].write_text(dump(self.policy))
        at["preference"].write_text(dump(self.preference))
        write_action_mask_catalogue(self.repo)
        return self

    def fakes(self, mp: pytest.MonkeyPatch, phi_loss: bool = False, loop=None) -> None:
        """build_phi faked (recording its config); `phi_loss`: PhiLoss faked too; `loop`:
        a replacement of `pretrain`. Patched in their modules and in run.py."""

        def build_phi(cfg):
            self.built.append(cfg)
            return PhiFloat32()

        patches = [(mcpref.models.perception.phi, "build_phi", build_phi)]
        if phi_loss:
            patches.append(
                (mcpref.models.perception.loss, "PhiLoss", lambda cfg: PhiLossStart())
            )
        if loop:
            patches.append((mcpref.training.pretrain.loop, "pretrain", loop))
        for module, name, value in patches:
            mp.setattr(module, name, value)
            mp.setattr(mcpref.models.factory, name, value, raising=False)
            mp.setattr(mcpref.training.pretrain.run, name, value, raising=False)

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
