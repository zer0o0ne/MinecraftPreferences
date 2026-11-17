"""Launch config validation and isolated effective config routing."""

import json

import pytest
import yaml
from runtime.play.conftest import PLAY

from mcpref.config import read_yaml
from mcpref.runtime.play import run
from mcpref.runtime.play.config import load_config
from mcpref.runtime.play.configs import create_directory, materialize


@pytest.mark.parametrize(
    "key,value",
    [
        ("mode", "random"),
        ("mode", {}),
        ("trajectories", 0),
        ("trajectories", True),
        ("trajectories", 1.5),
        ("device", False),
        ("device", "wrong"),
        ("out_dir", "../outside"),
        ("out_dir", "/tmp/out"),
        ("sampling", ""),
        ("rl_cycle", "../cycle.yaml"),
        ("extra", 1),
    ],
)
def test_invalid_config(tmp_path, key, value):
    path = tmp_path / "play.yaml"
    path.write_text(yaml.safe_dump(PLAY | {key: value}))
    with pytest.raises(ValueError):
        load_config(path)


def test_effective_rl_config_uses_runtime_paths_device_and_isolated_outputs(gameplay):
    root = gameplay.root
    pretrain_path = root / "configs/training/pretrain/config.yaml"
    pretrain_path.write_text(
        yaml.safe_dump(read_yaml(pretrain_path) | {"device": "cuda"})
    )
    before = {p: p.read_bytes() for p in (root / "configs").rglob("*.yaml")}
    raw = PLAY | {"mode": "rl"}
    gameplay.play_path.write_text(yaml.safe_dump(raw))
    before[gameplay.play_path] = gameplay.play_path.read_bytes()
    cfg = load_config(gameplay.play_path)
    out = create_directory(cfg)
    effective = materialize(cfg, out)
    sampling = read_yaml(root / effective.sampling)
    pretrain = read_yaml(root / sampling["pretrain"])
    diagnostics = read_yaml(root / pretrain["diagnostics"])
    rl = read_yaml(root / effective.rl_cycle)
    pairs = read_yaml(root / rl["rl_pairs"])
    assert rl["sampling"] == effective.sampling
    assert rl["device"] == pretrain["device"] == "cpu"
    assert pretrain["out_dir"] == "data/pretrain-scaled-init"
    assert [
        sampling["out_dir"],
        diagnostics["out_dir"],
        rl["out_dir"],
        pairs["out_dir"],
        pairs["history_dir"],
    ] == [
        "data/play-test/sampling",
        "data/play-test/diagnostics",
        "data/play-test/rl",
        "data/play-test/pairs",
        "data/play-test/history",
    ]
    assert read_yaml(out / "configs/play.yaml")["sampling"] == effective.sampling
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("existing", ["empty", "file", "symlink", "dangling"])
def test_reused_directory_rejected_before_env_without_changes(
    gameplay, monkeypatch, existing
):
    path = gameplay.root / PLAY["out_dir"]
    path.parent.mkdir(exist_ok=True)
    if existing == "empty":
        path.mkdir()
    elif existing == "file":
        path.write_bytes(b"keep")
    else:
        target = gameplay.root / ("missing" if existing == "dangling" else "configs")
        path.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(run, "Env", lambda cfg: pytest.fail("environment constructed"))
    with pytest.raises(FileExistsError, match="new gameplay out_dir"):
        run.run(gameplay.play_path)
    if existing == "empty":
        assert not list(path.iterdir())
    elif existing == "file":
        assert path.read_bytes() == b"keep"
    else:
        assert path.is_symlink()


@pytest.mark.parametrize("mode", ["pretrained", "rl"])
def test_incomplete_pretrain_rejected_before_env(gameplay, monkeypatch, mode):
    gameplay.play_path.write_text(yaml.safe_dump(PLAY | {"mode": mode}))
    pretrain = gameplay.root / "data/pretrain-scaled-init/run.json"
    pretrain.write_text('{"complete": false}')
    monkeypatch.setattr(run, "Env", lambda cfg: pytest.fail("environment constructed"))
    with pytest.raises(RuntimeError, match="not complete"):
        run.run(gameplay.play_path)
    metadata = json.loads((gameplay.root / PLAY["out_dir"] / "run.json").read_text())
    assert metadata["status"] == "failed" and metadata["completed_trajectories"] == 0
