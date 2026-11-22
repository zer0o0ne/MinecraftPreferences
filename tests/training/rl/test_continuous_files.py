"""Cross-cycle handover and immutable archived replay identities."""

import json

import pytest

from mcpref.infrastructure.runs.files import sha256
from mcpref.runtime.sampling.config import load_config as load_sampling
from mcpref.training.rl.config import load_config
from mcpref.training.rl.files import handover_from, validate_anchor_manifests
from training.pretrain.classical.data.index_run import repo_root
from training.rl.cycle_world import prepare
from training.rl.learning.helpers import write_json


def archived(root, cfg, *, complete=True):
    previous = root / cfg.out_dir / "000000"
    manifest = root / "data/sampling/shards/000000/anchors.json"
    write_json(manifest, {"version": 1, "stride": 4, "fingerprint": "a" * 64})
    write_json(
        previous / "run.json",
        {
            "complete": complete,
            "sha256": {"initial_model": "a" * 64, "anchors_manifest": sha256(manifest)},
        },
    )
    (previous / "model.pt").write_bytes(b"final")
    return previous, manifest


def test_handover_requires_completed_predecessor_and_final_path(tmp_path, repo):
    root, path = prepare(tmp_path, repo)
    cfg = load_config(path)
    previous, _ = archived(root, cfg)
    with repo_root(root):
        assert handover_from(cfg, 0, root / "unused") is None
        assert handover_from(cfg, 1, previous / "model.pt") == "a" * 64
        with pytest.raises(RuntimeError, match="previous completed"):
            handover_from(cfg, 1, previous / "other.pt")
        archived(root, cfg, complete=False)
        with pytest.raises(RuntimeError, match="previous completed"):
            handover_from(cfg, 1, previous / "model.pt")


def test_archived_anchor_manifest_changes_are_rejected(tmp_path, repo):
    root, path = prepare(tmp_path, repo)
    cfg = load_config(path)
    _, manifest = archived(root, cfg)
    with repo_root(root):
        sampling = load_sampling(root / cfg.sampling)
        validate_anchor_manifests(cfg, sampling, 1)
        value = json.loads(manifest.read_text()) | {"fingerprint": "b" * 64}
        manifest.write_text(json.dumps(value))
        with pytest.raises(RuntimeError, match="historical memory anchors"):
            validate_anchor_manifests(cfg, sampling, 1)
        manifest.unlink()
        with pytest.raises(RuntimeError, match="historical memory anchors"):
            validate_anchor_manifests(cfg, sampling, 1)
