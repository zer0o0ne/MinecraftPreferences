"""`check` (8)-(10) of specs/engine.md: MineStudio's simulator runs at >= min_rate."""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from mcpref.cli.failure import failed
from mcpref.engine.config import EngineConfig, load_config, use_engine

PROG = "mcpref.engine.check"


def step(sim, k: int, bar: tqdm) -> None:
    """(9): step k with a no-op action; an episode end raises; the bar advanced."""
    _, _, terminated, truncated, _ = sim.step(sim.noop_action())
    if terminated or truncated:
        raise RuntimeError(f"episode ended at step {k}")
    bar.update()


def rate(sim, cfg: EngineConfig) -> float:
    """(9), Notation rho: warmup_steps untimed steps, then steps timed ones per second."""
    sim.reset()
    total = cfg.warmup_steps + cfg.steps
    with tqdm(total=total, desc="check", unit="step", file=sys.stderr) as bar:
        for k in range(1, cfg.warmup_steps + 1):
            step(sim, k, bar)
        t0 = time.perf_counter()
        for k in range(cfg.warmup_steps + 1, total + 1):
            step(sim, k, bar)
        t1 = time.perf_counter()
    return cfg.steps / (t1 - t0)


def check(path: Path) -> None:
    """(8)-(10): the simulator stepped, closed in any case; rho below min_rate raises."""
    cfg = load_config(path)
    use_engine(cfg)
    from minestudio.simulator import MinecraftSim

    size = cfg.render_size
    sim = MinecraftSim(
        action_type="env", obs_size=size, render_size=size, seed=cfg.seed
    )
    try:
        rho = rate(sim, cfg)
    finally:
        sim.close()
    print(f"rate: {rho:.1f} ticks/s (min {cfg.min_rate})")
    if rho < cfg.min_rate:
        raise RuntimeError(f"rate {rho:.1f} ticks/s below min_rate {cfg.min_rate}")


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.engine.check <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        check(args.config)
    except Exception as error:  # noqa: BLE001 (specs/engine.md: any Exception)
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
