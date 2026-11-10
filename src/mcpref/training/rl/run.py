"""One CLI invocation: construct an Env, run one RL cycle, close the Env."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mcpref.config import REPO
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.runtime.env.config import load_config as load_env_config
from mcpref.runtime.env.wrapper import Env
from mcpref.runtime.sampling.config import load_config as load_sampling_config
from mcpref.training.rl.config import load_config
from mcpref.training.rl.cycle import run_cycle

PROG = "mcpref.training.rl.run"


def run(path: Path) -> None:
    """Run one cycle; a later process needs an externally matching world state."""
    cfg = load_config(path)
    sampling = load_sampling_config(REPO / cfg.sampling)
    env = Env(load_env_config(REPO / sampling.env))
    try:
        run_cycle(cfg, path, env)
    finally:
        env.close()


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.training.rl.run <config path>`; exit code."""
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
