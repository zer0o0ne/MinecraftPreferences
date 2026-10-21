"""Config, (8'), `main` of specs/preference-pool.md: the run: the rerun check and
run.json, (0) the indices, (H) the sessions, (P) and (W) the shards, the counts."""

import argparse
import dataclasses
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from mcpref import REPO, config_mapping, exact, integer, positive_int, relative
from mcpref.cli.failure import failed
from mcpref.clips.collect import OUTPUT_FIELDS as PERCEPTION_FIELDS
from mcpref.clips.collect import (
    PerceptionDataConfig,
    check_rerun,
    hashes,
    initial_pools,
    store_indices,
)
from mcpref.clips.collect import load_config as load_perception_config
from mcpref.clips.episode import Classify, Pipeline
from mcpref.clips.plan import pipeline
from mcpref.clips.run import gamma_of
from mcpref.clips.store import remove
from mcpref.pool.plan import resume, session_pools
from mcpref.pool.sessions import store_sessions
from mcpref.pool.shards import write_shards
from mcpref.runs.files import write_json

PROG = "mcpref.pool.run"
POSITIVE = ("head_bytes", "entries_per_shard", "tick_ms", "episodes", "per_episode")
POSITIVE += ("partners", "workers")
BOUNDED = {"head_workers": (1, 20), "n_clips": (2, math.inf), "seed": (0, 2**32 - 1)}
P_FIELDS = tuple(
    k for k in PERCEPTION_FIELDS if k not in ("per_episode", "n_clips", "seed")
)
OWN_FIELDS = ("entries_per_shard", "tick_ms", "tick_slack", "resetting_stats")
OWN_FIELDS += ("episodes", "per_episode", "n_clips", "partners", "cycles", "seed")


@dataclass(frozen=True)
class PreferencePoolConfig:
    """Notation: B, delta_t, eps, Z = head_bytes, tick_ms, tick_slack, resetting_stats;
    e, m_p, N_p, r, rho = episodes, per_episode, n_clips, partners, cycles."""

    perception: str
    head_bytes: int
    head_workers: int
    entries_per_shard: int
    tick_ms: int
    tick_slack: int | float
    resetting_stats: list[str]
    episodes: int
    per_episode: int
    n_clips: int
    partners: int
    cycles: int | float
    seed: int
    workers: int
    out_dir: str


def load_config(path: Path) -> PreferencePoolConfig:
    """Interface: the YAML file, validated, into PreferencePoolConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, PreferencePoolConfig)
    for k in ("perception", "out_dir"):
        relative(k, raw[k])
    for k in POSITIVE:
        positive_int(k, raw[k])
    for k, bounds in BOUNDED.items():
        integer(k, raw[k], *bounds)
    if not 0 <= exact("tick_slack", raw["tick_slack"]) < 1:
        raise ValueError(f"tick_slack {raw['tick_slack']!r} is not in [0, 1)")
    if not exact("cycles", raw["cycles"]) >= 0:
        raise ValueError(f"cycles {raw['cycles']!r} is not >= 0")
    stats = raw["resetting_stats"]
    if type(stats) is not list or any(type(x) is not str for x in stats):
        raise ValueError(f"resetting_stats {stats!r} is not a list of strings")
    return PreferencePoolConfig(**raw)


def run_header(cfg: PreferencePoolConfig, pipe: Pipeline, classifier_id: str) -> dict:
    """(8'): this config as written, then P's output fields as written in P's file; the
    hashes of perception-data (8); gamma_id."""
    config = dataclasses.asdict(cfg) | {k: getattr(pipe.cfg, k) for k in P_FIELDS}
    return {"config": config, "sha256": hashes(pipe), "classifier": classifier_id}


def collect(
    cfg: PreferencePoolConfig,
    p: PerceptionDataConfig,
    classify: Classify,
    classifier_id: str,
) -> None:
    """Interface: the check of (8') and run.json, staging removed, (0), (H), (P), (W),
    staging removed, then the counts added to run.json."""
    pipe = dataclasses.replace(pipeline(p, classify), out=REPO / cfg.out_dir)
    out, header = pipe.out, run_header(cfg, pipe, classifier_id)
    check_rerun(out / "run.json", header, P_FIELDS + OWN_FIELDS, "config.")
    write_json(out / "run.json", header)
    remove(out / "staging")
    with httpx.Client(headers={"x-ms-version": p.x_ms_version}) as client:
        store_indices(client, p, out / "indices")
        pools = initial_pools(p, out / "indices")
        store_sessions(cfg, p, client, out / "sessions", pools)
    bases = {s: basedir for s, (basedir, _) in pools.items()}
    sessions = session_pools(list(pools), out / "sessions")
    totals = write_shards(cfg, pipe, resume(out / "shards", cfg.seed, sessions), bases)
    remove(out / "staging")
    write_json(out / "run.json", header | totals.counts())


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.pool.run <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        p = load_perception_config(REPO / cfg.perception)
        collect(cfg, p, *gamma_of(p))
    except Exception as error:  # noqa: BLE001 (specs/preference-pool.md: any Exception)
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
