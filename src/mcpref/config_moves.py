"""Compatibility for pretraining config paths stored in existing artifacts."""

from pathlib import Path

_PREFIX = "configs/training/pretrain/"
MOVED = {
    _PREFIX + old: _PREFIX + new
    for old, new in (
        ("classical.yaml", "classical/config.yaml"),
        ("continuous.yaml", "continuous/config.yaml"),
        ("train-index.yaml", "classical/index/config.yaml"),
        ("shared-index.yaml", "classical/index/shared.yaml"),
        ("shared-labeled-index.yaml", "classical/index/shared-labeled.yaml"),
        ("continuous-index.yaml", "continuous/index.yaml"),
    )
}


def relocated_path(path: Path, root: Path) -> Path:
    """Resolve only missing known old paths; keep explicit overrides intact."""
    path = path.resolve()
    if path.exists() or not path.is_relative_to(root):
        return path
    target = MOVED.get(path.relative_to(root).as_posix())
    return root / target if target else path


def _legacy_references(payload: bytes) -> bytes:
    for old, new in MOVED.items():
        payload = payload.replace(new.encode(), old.encode())
    return payload


def pretrain_config_sha256(path: Path) -> str:
    """Keep historical config hashes for path-only pretraining relocations."""
    from mcpref.config import REPO, _config_sha256

    path = path.resolve()
    known = {*MOVED, *MOVED.values(), _PREFIX + "config.yaml"}
    legacy = path.is_relative_to(REPO) and path.relative_to(REPO).as_posix() in known
    return _config_sha256(path, _legacy_references if legacy else None)
