"""Real-engine smoke sequence of specs/env.md Tests."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from mcpref.infrastructure.cli.failure import failed
from mcpref.runtime.env.config import load_config
from mcpref.runtime.env.wrapper import Env

PROG = "mcpref.runtime.env.check"
ACTIONS = (
    "open_inventory",
    "close",
    "open_inventory",
    "left_click_1",
    "right_click_1",
    "shift_click_1",
    "double_click_1",
    "close",
    "attack",
    "use",
)


def check(path: Path) -> None:
    """Tests: the twelve work items, with close guaranteed by finally."""
    env = Env(load_config(path))
    try:
        env.reset()
        with tqdm(total=12, desc="env check", unit="action", file=sys.stderr) as bar:
            for name in ACTIONS:
                env.step(env.masks.names.index(name))
                bar.update()
            env._command("/kill")
            bar.update()
            env.step(env.masks.names.index(f"hotbar_{env.hotbar + 1}"))
            bar.update()
    finally:
        env.close()
    print("env: ok")


def main(argv: Sequence[str]) -> int:
    """Interface: one config path; any Exception reported by failed."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        check(args.config)
    except Exception as error:  # noqa: BLE001 (specs/env.md: any Exception)
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
