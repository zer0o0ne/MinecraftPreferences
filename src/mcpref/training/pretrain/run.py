"""(13) of specs/policy.md: `main` of the pretraining loop."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mcpref.config import REPO
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.models.factory import build_modules, check_lengths
from mcpref.models.perception.config import load_config as load_perception_config
from mcpref.models.policy.config import load_config as load_policy_config
from mcpref.models.preference.config import load_config as load_preference_config
from mcpref.training.pretrain.checks import index_run
from mcpref.training.pretrain.config import load_config
from mcpref.training.pretrain.loop import pretrain

PROG = "mcpref.training.pretrain.run"


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
    modules = build_modules(
        perception, masks, policy, preference, paired, cfg.device, cfg.latent_init_scale
    )
    pretrain(cfg, path, modules)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.training.pretrain.run <config path>`; exit code."""
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
