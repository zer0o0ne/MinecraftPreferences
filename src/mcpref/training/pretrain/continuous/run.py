"""Composition root of the continuous pretraining mode."""

import argparse
import sys
from pathlib import Path

from mcpref.config import REPO
from mcpref.gui.mask import load_config as mask_config
from mcpref.gui.mask import load_masks
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.models.factory import build_modules
from mcpref.models.perception.config import load_config as perception_config
from mcpref.models.policy.config import load_config as policy_config
from mcpref.models.preference.config import load_config as preference_config
from mcpref.training.pretrain.continuous.config import load_config
from mcpref.training.pretrain.continuous.data.reader import Index
from mcpref.training.pretrain.continuous.diagnostics.hooks import Diagnostics
from mcpref.training.pretrain.continuous.loop import pretrain


def run(path: Path) -> None:
    cfg = load_config(path)
    index = Index(REPO / cfg.index_dir)
    try:
        modules = build_modules(
            perception_config(REPO / cfg.perception),
            load_masks(mask_config(REPO / cfg.action_mask)),
            policy_config(REPO / cfg.policy),
            preference_config(REPO / cfg.preference),
            index.run["populations"]["train"]["N_P"] > 0,
            cfg.device,
            cfg.latent_init_scale,
            cfg.precision,
        )
        pretrain(cfg, path, modules, index, Diagnostics())
    finally:
        index.close()


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.config)
    except FAILURES as error:
        return failed("mcpref.training.pretrain.continuous.run", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
