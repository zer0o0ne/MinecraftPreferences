"""Mode selection and configuration ownership, independent of training internals."""

from dataclasses import asdict

import pytest
import yaml

from mcpref.training.pretrain import modes
from mcpref.training.pretrain.classical.config import load_config as classical
from mcpref.training.pretrain.continuous.config import load_config as continuous


def selection(mode):
    return {
        "mode": mode,
        "classical": {
            "config": "configs/classical.yaml",
            "index": "indexes/clips.yaml",
        },
        "continuous": {
            "config": "configs/continuous.yaml",
            "index": "indexes/streams.yaml",
        },
    }


@pytest.mark.parametrize("mode", ["classical", "continuous"])
def test_selected_paths(tmp_path, mode):
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selection(mode)))
    result = modes.load_selection(path)
    assert result.mode == mode
    assert asdict(result.selected) == selection(mode)[mode]


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "other"},
        {"mode": None},
        {"mode": []},
        {"continuous": "continuous.yaml"},
        {"classical": {"config": "../leaf.yaml", "index": "index.yaml"}},
        {"classical": {"config": "leaf.yaml"}},
        {"unused": 1},
    ],
)
def test_invalid_selection(tmp_path, change):
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selection("classical") | change))
    with pytest.raises(ValueError):
        modes.load_selection(path)


def test_clip_consumer_rejects_continuous_mode(tmp_path):
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selection("continuous")))
    with pytest.raises(RuntimeError, match="continuous perception stream adapter"):
        modes.classical_config_path(path)


def test_clip_consumer_resolves_classical_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(modes, "REPO", tmp_path)
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selection("classical")))
    assert modes.classical_config_path(path) == tmp_path / "configs/classical.yaml"
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("index_dir: data/index\n")
    assert modes.classical_config_path(leaf) == leaf


def test_repository_modes_share_model_and_optimizer_settings(repo):
    a = classical(repo / "configs/training/pretrain/classical/config.yaml")
    b = continuous(repo / "configs/training/pretrain/continuous/config.yaml")
    aa, bb = asdict(a), asdict(b)
    common = set(aa) & set(bb) - {"index_dir", "out_dir", "diagnostics"}
    assert {k: aa[k] for k in common} == {k: bb[k] for k in common}
    assert a.index_dir != b.index_dir and a.out_dir != b.out_dir
    assert "batch" not in bb and "tbptt_states" not in aa
    assert a.precision == b.precision == "fp32"
    assert b.publish_diagnostics is False
    assert b.activation_checkpointing is False


def test_materialized_continuous_leaf_still_rejects_classical_consumers(tmp_path):
    path = tmp_path / "pretrain.yaml"
    path.write_text("tbptt_states: 4\n")
    assert modes.selected_config(path) == ("continuous", path)
    with pytest.raises(RuntimeError, match="requires classical pretraining"):
        modes.classical_config_path(path)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_archive_retention_requires_a_positive_integer(repo, tmp_path, value):
    settings = asdict(
        continuous(repo / "configs/training/pretrain/continuous/config.yaml")
    )
    settings["betas"], settings["quantiles"] = (
        list(settings["betas"]),
        list(settings["quantiles"]),
    )
    settings["archive_keep_shards"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="archive_keep_shards"):
        continuous(path)


@pytest.mark.parametrize("name", ["publish_diagnostics", "activation_checkpointing"])
@pytest.mark.parametrize("value", [True, False, "false", 0, None])
def test_continuous_options_require_yaml_booleans(repo, tmp_path, name, value):
    settings = asdict(
        continuous(repo / "configs/training/pretrain/continuous/config.yaml")
    )
    settings["betas"] = list(settings["betas"])
    settings["quantiles"] = list(settings["quantiles"])
    settings[name] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    if type(value) is bool:
        assert getattr(continuous(path), name) is value
    else:
        with pytest.raises(ValueError, match=name):
            continuous(path)


def test_older_materialized_configs_keep_activation_checkpointing(repo, tmp_path):
    settings = asdict(
        continuous(repo / "configs/training/pretrain/continuous/config.yaml")
    )
    settings.pop("activation_checkpointing")
    settings["betas"] = list(settings["betas"])
    settings["quantiles"] = list(settings["quantiles"])
    path = tmp_path / "old-run.yaml"
    path.write_text(yaml.safe_dump(settings))
    assert continuous(path).activation_checkpointing is True
