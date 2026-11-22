"""Old collection fingerprints accept transport tuning and reject semantic drift."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mcpref.datasets.continuous import inputs
from mcpref.datasets.continuous.config import ContinuousConfig


@pytest.fixture
def resumed(monkeypatch, tmp_path):
    # This literal is the old configuration schema, before transfer_config existed.
    old_config = {
        "perception_config": "perception.yaml",
        "acquisition_plan": "plan",
        "legacy_provenance": "legacy.gz",
        "pair_clips_dir": "clips",
        "pair_labels_dir": "labels",
        "split_config": "split.yaml",
        "source_cache": "sources",
        "out_dir": "dataset",
        "shard_states": 7,
        "metadata_rows": 19,
        "max_shift": 3,
        "join_gap_ms": 1000,
        "disk_reserve_bytes": 100,
        "codec_threads": 2,
    }
    blobs = {
        "plan/manifest.json": b'{"frozen":"plan"}',
        "legacy.gz": b"frozen legacy source bytes",
        "split.yaml": b"validation_fraction: 0.2\n",
    }
    for name, content in blobs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source = {
        "config": {"width": 2, "height": 1, "workers": 1},
        "sha256": {"actions": "unchanged-action-config"},
        "classifier": "classifier-v1",
    }
    fingerprints = {
        "config": old_config,
        "source_config": source,
        "acquisition_plan": hashlib.sha256(blobs["plan/manifest.json"]).hexdigest(),
        "legacy_provenance": hashlib.sha256(blobs["legacy.gz"]).hexdigest(),
        "split_config": hashlib.sha256(blobs["split.yaml"]).hexdigest(),
        "labels": "frozen-label-hash",
        "sessions": "frozen-session-hash",
    }
    out = tmp_path / "dataset"
    out.mkdir()
    run = out / "run.json"
    run.write_text(json.dumps({"fingerprint": fingerprints, "complete": False}))
    monkeypatch.setattr(inputs, "REPO", tmp_path)
    monkeypatch.setattr(inputs, "run_header", lambda pipe, classifier_id: source)
    cfg = ContinuousConfig(**old_config, transfer_config="transfer.yaml")
    pipe = SimpleNamespace(
        cfg=SimpleNamespace(width=2, height=1), clips=SimpleNamespace(delta=6)
    )
    return SimpleNamespace(
        cfg=cfg,
        pipe=pipe,
        root=tmp_path,
        run=run,
        fingerprints=fingerprints,
        source=source,
    )


def header(case, cfg=None):
    return inputs.header(
        cfg or case.cfg,
        case.pipe,
        "classifier-v1",
        "frozen-label-hash",
        "frozen-session-hash",
    )


def test_old_pre_transfer_fingerprint_matches_exactly(resumed):
    result = header(resumed)
    assert result["fingerprint"] == resumed.fingerprints
    assert "transfer_config" not in result["fingerprint"]["config"]
    assert result["format"] == "mcpref-pretraining-streams-v1"
    assert (result["width"], result["height"], result["delta"]) == (2, 1, 6)


def test_transport_path_and_file_content_do_not_change_dataset_identity(resumed):
    original = resumed.root / "transfer.yaml"
    original.write_text("connections: 1\n")
    assert header(resumed)["fingerprint"] == resumed.fingerprints
    original.write_text("connections: 12\nblock_bytes: 1048576\n")
    assert header(resumed)["fingerprint"] == resumed.fingerprints
    alternate = resumed.root / "alternate-transfer.yaml"
    alternate.write_text("connections: 8\nblock_bytes: 262144\n")
    cfg = replace(resumed.cfg, transfer_config=alternate.name)
    assert header(resumed, cfg)["fingerprint"] == resumed.fingerprints
    assert json.loads(resumed.run.read_text())["fingerprint"] == resumed.fingerprints


@pytest.mark.parametrize(
    "key,value",
    [
        ("shard_states", 8),
        ("metadata_rows", 20),
        ("max_shift", 2),
        ("join_gap_ms", 999),
        ("disk_reserve_bytes", 101),
        ("codec_threads", 3),
        ("perception_config", "other-perception.yaml"),
        ("pair_clips_dir", "other-clips"),
        ("pair_labels_dir", "other-labels"),
        ("source_cache", "other-sources"),
    ],
)
def test_existing_config_fields_remain_pinned(resumed, key, value):
    before = resumed.run.read_bytes()
    with pytest.raises(ValueError, match="inputs/config changed"):
        header(resumed, replace(resumed.cfg, **{key: value}))
    assert resumed.run.read_bytes() == before


@pytest.mark.parametrize("name", ["plan/manifest.json", "legacy.gz", "split.yaml"])
def test_frozen_input_bytes_still_gate_resume(resumed, name):
    path = resumed.root / name
    path.write_bytes(path.read_bytes() + b"\n")
    before = resumed.run.read_bytes()
    with pytest.raises(ValueError, match="inputs/config changed"):
        header(resumed)
    assert resumed.run.read_bytes() == before


@pytest.mark.parametrize("field", ["labels", "sessions"])
def test_annotation_identity_is_still_pinned(resumed, field):
    arguments = {"labels": "frozen-label-hash", "sessions": "frozen-session-hash"}
    arguments[field] = "changed-hash"
    before = resumed.run.read_bytes()
    with pytest.raises(ValueError, match="inputs/config changed"):
        inputs.header(resumed.cfg, resumed.pipe, "classifier-v1", **arguments)
    assert resumed.run.read_bytes() == before


def test_source_configuration_is_not_removed_from_fingerprint(resumed):
    resumed.source["config"]["workers"] = 2
    before = resumed.run.read_bytes()
    with pytest.raises(ValueError, match="inputs/config changed"):
        header(resumed)
    assert resumed.run.read_bytes() == before


def test_relocated_split_preserves_recorded_identity_and_still_checks_content(resumed):
    old = "configs/training/pretrain/train-index.yaml"
    new = "configs/training/pretrain/classical/index/config.yaml"
    resumed.fingerprints["config"]["split_config"] = old
    resumed.run.write_text(json.dumps({"fingerprint": resumed.fingerprints}))
    path = resumed.root / new
    path.parent.mkdir(parents=True)
    (resumed.root / "split.yaml").rename(path)
    cfg = replace(resumed.cfg, split_config=new)
    assert header(resumed, cfg)["fingerprint"] == resumed.fingerprints
    path.write_text("validation_fraction: 0.3\n")
    before = resumed.run.read_bytes()
    with pytest.raises(ValueError, match="inputs/config changed"):
        header(resumed, cfg)
    assert resumed.run.read_bytes() == before
