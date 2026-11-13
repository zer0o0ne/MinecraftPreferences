"""Repository-relative YAML inheritance and removal."""

import hashlib
from pathlib import Path

import pytest
import yaml

from mcpref import config


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_extends_deep_merges_in_order_and_remove_preserves_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO", tmp_path)
    write(
        tmp_path,
        "base.yaml",
        "groups:\n"
        "  first: {lr: 1}\n"
        "  remove_me: {lr: 2}\n"
        "  last: {lr: 3}\n"
        "nested: {left: 1, right: 2}\n"
        "items: [1, 2]\n"
        "value: 1\n",
    )
    write(tmp_path, "other.yaml", "nested: {left: 3, extra: 4}\nvalue: 2\n")
    write(
        tmp_path,
        "leaf.yaml",
        "extends: [base.yaml, other.yaml]\n"
        "remove: [groups.remove_me]\n"
        "nested: {right: 5}\n"
        "items: [6]\n",
    )
    result = config.read_yaml(tmp_path / "leaf.yaml")
    assert result == {
        "groups": {"first": {"lr": 1}, "last": {"lr": 3}},
        "nested": {"left": 3, "right": 5, "extra": 4},
        "items": [6],
        "value": 2,
    }
    assert list(result["groups"]) == ["first", "last"]
    assert "extends" not in result and "remove" not in result


def test_starting_configs_resolve_to_their_previous_values():
    pretrain = config.read_yaml(config.REPO / "configs/training/pretrain/config.yaml")
    rl = config.read_yaml(config.REPO / "configs/training/rl/cycle.yaml")
    assert "phi_dual" in pretrain["groups"] and "metra_dual" not in pretrain["groups"]
    assert "metra_dual" in rl["groups"] and "phi_dual" not in rl["groups"]
    assert list(pretrain["groups"]).index("phi_dual") < list(pretrain["groups"]).index(
        "pi"
    )
    assert list(rl["groups"]).index("metra_dual") < list(rl["groups"]).index("pi")
    assert pretrain["betas"] == rl["betas"] == [0.9, 0.999]
    assert pretrain["retrain"] is True and rl["retrain"] is True
    assert pretrain["gradient_diagnostics"] is False
    assert rl["gradient_diagnostics"] is False
    assert pretrain["w_P"] == rl["w_P"] == 1.0
    pretrain_raw = yaml.safe_load(
        (config.REPO / "configs/training/pretrain/config.yaml").read_text()
    )
    rl_raw = yaml.safe_load(
        (config.REPO / "configs/training/rl/cycle.yaml").read_text()
    )
    assert pretrain_raw["retrain"] is True and rl_raw["retrain"] is True
    assert pretrain_raw["w_P"] == rl_raw["w_P"] == 1.0
    assert rl["items_per_shard"] == 256
    policy = config.read_yaml(config.REPO / "configs/models/policy.yaml")
    preference = config.read_yaml(config.REPO / "configs/models/preference.yaml")
    for cfg in (policy, preference):
        assert {"history", "width", "heads", "hidden", "rope_base", "norm_eps"} <= set(
            cfg
        )
        assert "extends" not in cfg
    assert policy["layers"] == 4
    assert preference["layers"] == ["S", "X", "S", "X"]


def test_inheritance_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO", tmp_path)
    write(tmp_path, "base.yaml", "groups: {present: 1}\n")
    write(tmp_path, "a.yaml", "extends: [b.yaml]\n")
    write(tmp_path, "b.yaml", "extends: [a.yaml]\n")
    with pytest.raises(ValueError, match="config extends cycle"):
        config.read_yaml(tmp_path / "a.yaml")
    write(tmp_path, "unknown.yaml", "extends: [base.yaml]\nremove: [groups.missing]\n")
    with pytest.raises(ValueError, match="unknown remove path"):
        config.read_yaml(tmp_path / "unknown.yaml")
    write(tmp_path, "invalid.yaml", "extends: base.yaml\n")
    with pytest.raises(ValueError, match="extends must be a list"):
        config.read_yaml(tmp_path / "invalid.yaml")
    raw = yaml.safe_load((tmp_path / "a.yaml").read_text())
    assert raw == {"extends": ["b.yaml"]}


def test_config_hash_includes_parents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO", tmp_path)
    write(tmp_path, "base.yaml", "value: 1\n")
    write(tmp_path, "plain.yaml", "value: 1\n")
    plain = tmp_path / "plain.yaml"
    assert config.config_sha256(plain) == hashlib.sha256(plain.read_bytes()).hexdigest()
    write(tmp_path, "leaf.yaml", "extends: [base.yaml]\nvalue: 2\n")
    first = config.config_sha256(tmp_path / "leaf.yaml")
    assert first == config.config_sha256(tmp_path / "leaf.yaml")
    write(tmp_path, "base.yaml", "value: 3\n")
    assert config.config_sha256(tmp_path / "leaf.yaml") != first
