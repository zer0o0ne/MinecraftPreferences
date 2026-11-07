"""(R) and (I) input checks of specs/vlm-labels.md."""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from mcpref import REPO
from mcpref.recognition.stored import shard_bar
from mcpref.runs import files

if TYPE_CHECKING:
    from mcpref.preferences.labels import VlmLabelsConfig


def checks(cfg: "VlmLabelsConfig") -> dict:
    run = files.finished_run(REPO / cfg.pool_dir, cfg.pool_dir, "clips")
    states = run["config"]["states"]
    if (cfg.frames - 1) * cfg.stride > states - 1:
        raise RuntimeError(
            f"frames and stride exceed the {states} states of {cfg.pool_dir}"
        )
    return run


def inputs(cfg: "VlmLabelsConfig", path: Path, run: dict, K: int) -> dict:
    with shard_bar("label inputs", 2 * K, 0, 2 * K, 0, "shard") as update:
        fields = files.clip_fields(run)
        pool = files.fingerprint(cfg.pool_dir, "records.json", fields, update)
        comparisons = files.fingerprint(cfg.pool_dir, "comparisons.json", None, update)
    sha = {"config": files.sha256(path), "vlm": files.sha256(REPO / cfg.vlm)}
    sha |= {"pool": pool, "comparisons": comparisons}
    files.check_fingerprints(REPO / cfg.out_dir / "run.json", sha, cfg.out_dir)
    header = {"config": yaml.safe_load(path.read_text()), "sha256": sha}
    files.write_json(REPO / cfg.out_dir / "run.json", header)
    return header
