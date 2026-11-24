"""Build the selected mode's index, then launch its pretraining loop."""

import argparse
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

from mcpref.config import REPO
from mcpref.config_moves import relocated_path
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.runs.lock import exclusive_run
from mcpref.training.pretrain.modes import load_selection

PROG = "mcpref.training.pretrain.run"


def run(path: Path) -> None:
    selection = load_selection(path)
    config_path = relocated_path(REPO / selection.selected.config, REPO)
    index_path = relocated_path(REPO / selection.selected.index, REPO)
    prefix = f"mcpref.training.pretrain.{selection.mode}"
    index = import_module(f"{prefix}.data.index")
    trainer = import_module(f"{prefix}.run")
    config = trainer.load_config(config_path)
    index_config = index.load_config(index_path)
    if (REPO / config.index_dir).resolve() != (REPO / index_config.out_dir).resolve():
        raise ValueError("Selected index output does not match pretrain index_dir")
    if (
        selection.mode == "continuous"
        and relocated_path(REPO / index_config.training_config, REPO) != config_path
    ):
        raise ValueError(
            "Continuous index training_config does not match the selected trainer config"
        )
    guard = (
        exclusive_run(REPO / config.out_dir)
        if selection.mode == "continuous"
        else nullcontext()
    )
    with guard:
        index.run(index_config, index_path)
        trainer.run(config_path)


def main(argv: Sequence[str]) -> int:
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
