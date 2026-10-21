"""`main` of specs/perception-data.md: the classifier of specs/gui-classifier.md, then the
run of `collect`; one-line message on stderr and exit 1 on any exception (9)."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mcpref import REPO
from mcpref.cli.failure import failed
from mcpref.clips.collect import PerceptionDataConfig, collect, load_config
from mcpref.clips.episode import Classify
from mcpref.recognition import classifier

PROG = "mcpref.clips.run"


def gamma_of(cfg: PerceptionDataConfig) -> tuple[Classify, str]:
    """Interface `main`: gamma and gamma_id by load_classifier of specs/gui-classifier.md
    (the config at REPO / classifier_config, the screens, W, H)."""
    gamma_cfg = classifier.load_config(REPO / cfg.classifier_config)
    return classifier.load_classifier(gamma_cfg, cfg.screens, cfg.width, cfg.height)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.clips.run <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        collect(cfg, *gamma_of(cfg))
    except Exception as error:  # noqa: BLE001 (specs/perception-data.md: any Exception)
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
