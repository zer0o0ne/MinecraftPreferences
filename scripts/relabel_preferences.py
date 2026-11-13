"""One-off rebuild of saved Qwen votes into a separate corrected label dataset."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from mcpref import config
from mcpref.datasets.preferences import acyclic
from mcpref.diagnostics.core.config import load_config as load_diagnostics
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.shards.store import write_file


def save_config(path: Path, value: dict) -> None:
    """Keep effective configs stable across interrupted migrations."""
    data = yaml.safe_dump(value, sort_keys=False).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"migration config changed: {path}")
    else:
        write_file(path, data)


def destination(path: Path, output: str, cfg: acyclic.AcyclicConfig) -> Path:
    """Reject destinations overlapping the source data or existing diagnostics."""
    config.relative("output", output)
    root = config.REPO.resolve()
    out = (root / output).resolve()
    diag_path = root / cfg.diagnostics
    diag = load_diagnostics(diag_path)
    sources = [
        path,
        diag_path,
        root / cfg.votes_dir,
        root / cfg.out_dir,
        root / diag.out_dir,
    ]
    if not out.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    for source in sources:
        source = source.resolve()
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError(f"output overlaps input: {source}")
    if (
        out.exists()
        and not (out / "acyclic.yaml").is_file()
        and any(p.name != "acyclic.yaml.part" for p in out.iterdir())
    ):
        raise ValueError(f"output is not a migration directory: {out}")
    return out


def rebuild(path: Path, output: str) -> None:
    """Reuse ordinary graph correction, diagnostics and shard resume without Qwen."""
    cfg = acyclic.load_config(path)
    acyclic.checks(cfg, path)
    out = destination(path, output, cfg)
    written = out.relative_to(config.REPO.resolve()).as_posix()
    effective = config.read_yaml(path) | {
        "out_dir": written,
        "diagnostics": f"{written}/diagnostics.yaml",
    }
    diagnostics = config.read_yaml(config.REPO / cfg.diagnostics) | {
        "out_dir": f"{written}/diagnostics",
    }
    effective_path = out / "acyclic.yaml"
    save_config(effective_path, effective)
    save_config(out / "diagnostics.yaml", diagnostics)
    acyclic.run(acyclic.load_config(effective_path), effective_path)
    counts = json.loads((out / "run.json").read_text())["counts"]
    print(json.dumps({"out_dir": written, "counts": counts}))
    print("Rebuild the training index with pair_labels_dir=" + written, file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="existing acyclic YAML config")
    parser.add_argument(
        "output", help="new label directory, relative to the repository"
    )
    args = parser.parse_args(argv)
    try:
        rebuild(args.config, args.output)
    except FAILURES as error:
        return failed("relabel_preferences", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
