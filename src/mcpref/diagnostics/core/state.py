"""Persistent run.json state of producer-owned diagnostics."""

import json
from pathlib import Path

import yaml

from mcpref.config import REPO, config_sha256
from mcpref.diagnostics.core.artifacts import write_accepted_index
from mcpref.diagnostics.core.config import DiagnosticsConfig
from mcpref.infrastructure.runs.files import write_json


def load_state(path: Path, cfg: DiagnosticsConfig) -> dict:
    out_dir = REPO / cfg.out_dir
    run_path = out_dir / "run.json"
    config = yaml.safe_load(path.read_text())
    fingerprints = {"config": config_sha256(path), "inputs": {}}
    if run_path.exists():
        state = json.loads(run_path.read_bytes())
        if (
            state["config"] != config
            or state["sha256"]["config"] != fingerprints["config"]
        ):
            raise RuntimeError("diagnostics config changed since the run started")
        return state
    return {
        "config": config,
        "sha256": fingerprints,
        "channels": {},
        "complete": False,
        "counts": {},
    }


def save_state(out_dir: Path, state: dict, *, index: bool = True) -> None:
    state["counts"] = {
        channel: {"events": len(value["events"]), "records": value["records"]}
        for channel, value in state["channels"].items()
    }
    if index:
        write_accepted_index(out_dir)
    write_json(out_dir / "run.json", state)
