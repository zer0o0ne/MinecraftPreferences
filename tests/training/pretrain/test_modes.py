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
    a = classical(repo / "configs/training/pretrain/classical.yaml")
    b = continuous(repo / "configs/training/pretrain/continuous.yaml")
    aa, bb = asdict(a), asdict(b)
    common = set(aa) & set(bb) - {"index_dir", "out_dir", "diagnostics"}
    assert {k: aa[k] for k in common} == {k: bb[k] for k in common}
    assert a.index_dir != b.index_dir and a.out_dir != b.out_dir
    assert "batch" not in bb and "tbptt_states" not in aa
