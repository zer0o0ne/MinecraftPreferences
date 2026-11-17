"""Run configured pretrained+MCTS or RL gameplay in one live world."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.runs.files import write_json
from mcpref.runtime.env.config import load_config as load_env
from mcpref.runtime.env.wrapper import Env
from mcpref.runtime.play.config import PlayConfig, load_config
from mcpref.runtime.play.configs import create_directory, materialize
from mcpref.runtime.sampling.config import load_config as load_sampling
from mcpref.runtime.sampling.runner import Sampler
from mcpref.training.rl.config import load_config as load_rl
from mcpref.training.rl.cycle import run_cycle


def _play(cfg: PlayConfig, root: Path, metadata: dict) -> None:
    sampling = load_sampling(REPO / cfg.sampling)
    rl = load_rl(REPO / cfg.rl_cycle) if cfg.mode == "rl" else None
    env = Env(load_env(REPO / sampling.env))
    try:
        if cfg.mode == "pretrained":
            sampler = Sampler(sampling, env)
            sampler.reset()
        for _ in tqdm(
            range(cfg.trajectories), desc="gameplay", unit="trajectory", file=sys.stderr
        ):
            if cfg.mode == "pretrained":
                sampler.sample()
            else:
                run_cycle(rl, REPO / cfg.rl_cycle, env)
            metadata["completed_trajectories"] += 1
            write_json(root / "run.json", metadata)
    finally:
        env.close()


def run(path: Path) -> Path:
    cfg = load_config(path)
    root = create_directory(cfg)
    metadata = {
        "config": asdict(cfg),
        "completed_trajectories": 0,
        "status": "running",
    }
    try:
        write_json(root / "run.json", metadata)
        effective = materialize(cfg, root)
        metadata["config"] = asdict(effective)
        write_json(root / "run.json", metadata)
        _play(effective, root, metadata)
    except BaseException as error:
        metadata |= {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        write_json(root / "run.json", metadata)
        raise
    metadata["status"] = "complete"
    write_json(root / "run.json", metadata)
    return root


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, nargs="?", default=REPO / "configs/runtime/play.yaml"
    )
    args = parser.parse_args(argv)
    try:
        output = run(args.config)
    except FAILURES as error:
        return failed(parser.prog, error)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
