"""(R) of specs/pretrain.md: index, inputs' fingerprints and run.json checks."""

from pathlib import Path

import yaml

from mcpref.config import REPO, config_sha256
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.infrastructure.progress import shard_bar
from mcpref.infrastructure.runs.files import (
    check_fingerprints,
    clip_fields,
    fingerprint,
    finished_run,
    sha256,
    write_json,
)
from mcpref.infrastructure.shards.store import complete_shards
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import CONFIGS, PretrainConfig
from mcpref.training.pretrain.data.index import FINGERPRINTS
from mcpref.training.pretrain.data.items import PAIRS, RECORDS

MODULES = ("phi", "phi_loss", "pi", "pi_loss", "P", "P_loss")


def index_run(cfg: PretrainConfig) -> dict:
    """(R) 1, 2: {X}/run.json complete, N >= 1."""
    run = finished_run(REPO / cfg.index_dir, cfg.index_dir, "complete")
    if run["counts"]["train"]["clips"] + run["counts"]["train"]["pairs"] < 1:
        raise RuntimeError(f"{cfg.index_dir} has no training item")
    return run


def given(cfg: PretrainConfig, run: dict, modules: Modules) -> Modules:
    """(R) 3: the required modules given (P, P_loss iff N_R >= 1); in module order."""
    names = list(MODULES[: 6 if run["counts"]["train"]["pairs"] else 4])
    if set(modules) != set(names):
        raise RuntimeError(
            f"{cfg.index_dir} needs the modules {names}, given {list(modules)}"
        )
    return {name: modules[name] for name in names}


def check_inputs(cfg: PretrainConfig, run: dict) -> None:
    """(R) 5: C (and Q_c) complete; the fingerprints of C, Q_c, Q_l (bar `pretrain
    inputs`) equal those of {X}/run.json."""
    c = run["config"]
    paired = c["pair_clips_dir"] is not None
    clips = [c["clips_dir"], *([c["pair_clips_dir"]] if paired else [])]
    fields = [clip_fields(finished_run(REPO / d, d, "clips")) for d in clips]
    datasets = [(d, RECORDS, f) for d, f in zip(clips, fields)]
    datasets += [(c["pair_labels_dir"], PAIRS, None)] if paired else []
    n = sum(complete_shards(REPO / d / "shards") for d, _, _ in datasets)
    with shard_bar("pretrain inputs", n, 0, n, 0, "shard") as update:
        found = [fingerprint(d, file, f, update) for d, file, f in datasets]
    for name, value in zip(FINGERPRINTS[1:], found):
        if run["sha256"][name] != value:
            raise RuntimeError(f"{name} changed since {cfg.index_dir} was built")


def header(cfg: PretrainConfig, path: Path) -> dict:
    """(R) 6: check_fingerprints of an existing {out_dir}/run.json (the config, the
    index, the module configs, the screen catalogue); run.json written."""
    out = REPO / cfg.out_dir / "run.json"
    sha = {
        "config": config_sha256(path),
        "index": sha256(REPO / cfg.index_dir / "run.json"),
    }
    sha |= {k: config_sha256(REPO / getattr(cfg, k)) for k in CONFIGS}
    sha["screens"] = sha256(REPO / load_mask_config(REPO / cfg.action_mask).screens)
    check_fingerprints(out, sha, cfg.out_dir)
    head = {"config": yaml.safe_load(path.read_text()), "sha256": sha}
    write_json(out, head)
    return head
