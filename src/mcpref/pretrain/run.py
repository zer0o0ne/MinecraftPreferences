"""(13) of specs/policy.md: `main` of the pretraining loop - the configs, the clip
length checks, the modules phi, pi (and P), then `pretrain`."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.gui.mask import Masks, load_masks
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.perception.config import PerceptionConfig
from mcpref.perception.config import load_config as load_perception_config
from mcpref.perception.loss import PhiLoss, latent_metric
from mcpref.perception.phi import build_phi
from mcpref.policy.config import PolicyConfig
from mcpref.policy.config import load_config as load_policy_config
from mcpref.policy.loss import PolicyLoss
from mcpref.policy.network import Policy
from mcpref.preferences.config import PreferenceConfig
from mcpref.preferences.config import load_config as load_preference_config
from mcpref.preferences.loss import PreferenceLoss
from mcpref.preferences.network import Preference
from mcpref.pretrain.checks import index_run
from mcpref.pretrain.config import load_config
from mcpref.pretrain.loop import pretrain

PROG = "mcpref.pretrain.run"


def check_lengths(
    states: int,
    policy: PolicyConfig,
    preference: PreferenceConfig,
    paired: bool,
    index_dir: str,
) -> None:
    """(13): L >= 3, L_pi = L, with pairs L_P <= L, in this order."""
    if states < 3:
        raise RuntimeError(
            f"{index_dir} has clips of {states} states, the policy needs at least 3"
        )
    if policy.history != states:
        raise RuntimeError(
            f"the policy history {policy.history} differs from the clip length "
            f"{states} of {index_dir}"
        )
    if paired and preference.history > states:
        raise RuntimeError(
            f"the preference history {preference.history} exceeds the clip length "
            f"{states} of {index_dir}"
        )


def build_modules(
    perception: PerceptionConfig,
    masks: Masks,
    policy: PolicyConfig,
    preference: PreferenceConfig,
    paired: bool,
    device: str,
) -> dict[str, torch.nn.Module]:
    """(13): phi, phi_loss, pi, pi_loss (with pairs P, P_loss), in module order, each
    moved to `device`."""
    D, target = len(perception.alpha), torch.device(device)
    modules = {"phi": build_phi(perception).to(target)}
    modules["phi_loss"] = PhiLoss(perception).to(target)
    modules["pi"] = Policy(policy, D, masks.names).to(target)
    metric = latent_metric(perception)
    modules["pi_loss"] = PolicyLoss(policy, masks, metric).to(target)
    if paired:
        modules["P"] = Preference(preference, D).to(target)
        modules["P_loss"] = PreferenceLoss().to(target)
    return modules


def run(path: Path) -> None:
    """(13): the PretrainConfig at `path`, the configs it names, index_run, the checks,
    the modules, then the loop."""
    cfg = load_config(path)
    perception = load_perception_config(REPO / cfg.perception)
    masks = load_masks(load_mask_config(REPO / cfg.action_mask))
    policy = load_policy_config(REPO / cfg.policy)
    preference = load_preference_config(REPO / cfg.preference)
    index = index_run(cfg)
    paired = index["counts"]["train"]["pairs"] >= 1
    check_lengths(index["states"], policy, preference, paired, cfg.index_dir)
    modules = build_modules(perception, masks, policy, preference, paired, cfg.device)
    pretrain(cfg, path, modules)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.pretrain.run <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.config)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
