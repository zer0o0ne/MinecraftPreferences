"""One CLI invocation: construct an Env, run one RL cycle, close the Env."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mcpref import REPO
from mcpref.cli.failure import FAILURES, failed
from mcpref.env.config import load_config as load_env_config
from mcpref.env.wrapper import Env
from mcpref.rl.config import load_config
from mcpref.rl.cycle import run_cycle
from mcpref.sampling.config import load_config as load_sampling_config

PROG = "mcpref.rl.run"


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
    """`python -m mcpref.rl.run <config path>`; exit code."""
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
