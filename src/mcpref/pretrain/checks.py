"""(R) of specs/pretrain.md: the index, the modules and their groups, the inputs'
fingerprints, then {out_dir}/run.json."""

from pathlib import Path

import torch
import yaml

from mcpref import REPO
from mcpref.pretrain.checkpoints import Modules
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.index import FINGERPRINTS, clip_fields
from mcpref.pretrain.items import PAIRS, RECORDS, fingerprint
from mcpref.recognition.stored import shard_bar
from mcpref.runs.files import check_fingerprints, finished_run, sha256, write_json
from mcpref.shards.store import complete_shards

MODULES = ("phi", "phi_loss", "pi", "pi_loss", "P", "P_loss")
Groups = dict[str, list[torch.nn.Parameter]]
"""Notation: the groups {name: parameters} in group order."""


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


def declared(modules: Modules) -> Groups:
    """(R) 4: the groups by module, then by declaration; a name twice aborts."""
    groups = {}
    for module in modules.values():
        for q, params in module.groups().items():
            if q in groups:
                raise RuntimeError(f"group {q} declared twice")
            groups[q] = params
    return groups


def check_groups(cfg: PretrainConfig, modules: Modules) -> Groups:
    """(R) 4: names once, each parameter in one group, every parameter in a group, the
    names those of the config."""
    groups, owner = declared(modules), {}
    for q, params in groups.items():
        for p in params:
            if id(p) in owner:
                raise RuntimeError(
                    f"a parameter is twice in the group {q}"
                    if owner[id(p)] == q
                    else f"a parameter is in the groups {owner[id(p)]} and {q}"
                )
            owner[id(p)] = q
    for name, module in modules.items():
        if any(id(p) not in owner for p in module.parameters()):
            raise RuntimeError(f"{name} has parameters outside the groups")
    names, configured = list(groups), list(cfg.groups)
    if set(names) != set(configured):
        raise RuntimeError(
            f"the modules declare the groups {names}, the config {configured}"
        )
    return groups


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
    """(R) 6: check_fingerprints of an existing {out_dir}/run.json; run.json written."""
    out = REPO / cfg.out_dir / "run.json"
    sha = {"config": sha256(path), "index": sha256(REPO / cfg.index_dir / "run.json")}
    check_fingerprints(out, sha, cfg.out_dir)
    head = {"config": yaml.safe_load(path.read_text()), "sha256": sha}
    write_json(out, head)
    return head
