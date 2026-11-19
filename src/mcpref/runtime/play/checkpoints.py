"""Freeze the selected initial artifact against concurrent training checkpoint pruning."""

import shutil
from pathlib import Path

from mcpref.infrastructure.runs.files import sha256, write_json
from mcpref.infrastructure.shards.store import make_dirs, part_path, publish, write_file


def snapshot(source: Path, directory: Path) -> None:
    make_dirs(directory)
    target = directory / source.name
    staged = part_path(target)
    shutil.copyfile(source, staged)
    publish(staged, target)
    if source.name == "model.pt":
        write_file(directory / "run.json", (source.parent / "run.json").read_bytes())
    write_json(
        directory / "source.json",
        {"path": str(source), "sha256": sha256(target)},
    )
