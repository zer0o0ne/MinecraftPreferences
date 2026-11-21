"""Verify committed fragments and finish interrupted source cleanup on resume."""

import json
from pathlib import Path

from mcpref.config import REPO
from mcpref.datasets.continuous.config import digest
from mcpref.infrastructure.runs.files import write_json
from mcpref.infrastructure.shards.store import shard_path


def seal(directory: Path):
    files = {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    write_json(directory / "state.json", {"files": files})


def verify_files(directory: Path):
    state = json.loads((directory / "state.json").read_text())
    for name, checksum in state["files"].items():
        if digest(directory / name) != checksum:
            raise ValueError(f"Committed file checksum differs: {directory / name}")


def cleanup(cfg, summary):
    for source in summary["sources"]:
        target = REPO / cfg.source_cache / source["piece"]["piece_id"]
        if source["reason"] is None:
            write_json(
                target / "published.json",
                {"dataset": cfg.out_dir, "fragment_shard": summary["manifest_index"]},
            )
            (target / "source.mp4").unlink(missing_ok=True)


def recover(cfg, out, k):
    directory = shard_path(out / "fragments", k)
    verify_files(directory)
    summary = json.loads((directory / "summary.json").read_text())
    for shard in range(summary["first_shard"], summary["next_shard"]):
        verify_files(shard_path(out / "shards", shard))
    cleanup(cfg, summary)
    return summary
