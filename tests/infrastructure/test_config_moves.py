"""Known path aliases preserve pretraining identity, not arbitrary config edits."""

import hashlib

import pytest

from mcpref import config
from mcpref.config_moves import MOVED, pretrain_config_sha256


def test_relocated_inheritance_keeps_old_hash_and_checks_all_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "REPO", tmp_path)
    prefix = "configs/training/pretrain/"
    parent_name, leaf_name = (
        prefix + "shared-index.yaml",
        prefix + "shared-labeled-index.yaml",
    )
    parent, leaf = tmp_path / parent_name, tmp_path / leaf_name
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"seed: 7\n")
    leaf.write_text(f"extends:\n  - {parent_name}\nlabels: labeled\n")
    original = hashlib.sha256(
        leaf.read_bytes() + b"\0" + parent.read_bytes()
    ).hexdigest()
    moved_parent, moved_leaf = (
        tmp_path / MOVED[parent_name],
        tmp_path / MOVED[leaf_name],
    )
    moved_parent.parent.mkdir(parents=True)
    parent.rename(moved_parent)
    leaf.rename(moved_leaf)
    moved_leaf.write_text(
        moved_leaf.read_text().replace(parent_name, MOVED[parent_name])
    )
    for path in (leaf, moved_leaf):
        assert config.read_yaml(path) == {"seed": 7, "labels": "labeled"}
        assert pretrain_config_sha256(path) == original
        assert config.config_sha256(path) != original
    moved_parent.write_bytes(b"seed: 8\n")
    assert pretrain_config_sha256(moved_leaf) != original


def test_existing_old_file_and_outside_repository_are_not_redirected(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    monkeypatch.setattr(config, "REPO", root)
    old, new = next(iter(MOVED.items()))
    for name, value in ((old, 1), (new, 2)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"value: {value}\n")
    assert config.read_yaml(root / old) == {"value": 1}
    assert config.read_yaml(root / new) == {"value": 2}
    with pytest.raises(FileNotFoundError):
        config.read_yaml(tmp_path / old)


def test_unrelated_config_hash_is_not_canonicalized(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO", tmp_path)
    path = tmp_path / "custom.yaml"
    path.write_text(f"config: {next(iter(MOVED.values()))}\n")
    assert pretrain_config_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
