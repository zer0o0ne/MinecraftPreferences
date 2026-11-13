"""Saved-log viewer oracles use persisted values, with no checkpoints or model calls."""

import json

import pytest

from mcpref.diagnostics.saved import iter_saved_runs
from mcpref.infrastructure.shards.store import write_shard


def shard(directory, k, files):
    with write_shard(directory, k) as staged:
        for filename, value in files.items():
            (staged / filename).write_text(json.dumps(value))


def test_pretrain_preserves_scalars_and_saved_window_values(tmp_path):
    step = {
        "step": 7,
        "phi": 2.0,
        "pi": 3.0,
        "P": None,
        "loss": 37.0,
        "lr": {"phi.body": 0.002},
        "counts": {"pairs": 0},
        "gradients": {"global": {"norm": 4.0, "active": True}},
        "outputs": {"P_inc": None, "pi_action": {"mean_confidence": 0.7}},
        "components": {"P": None},
        "version": 99,
    }
    window = {
        "version": 1,
        "first_step": 1,
        "last_step": 7,
        "partial": True,
        "losses": {"phi": 93.0, "P": None},
        "coverage": {"losses": {"P": 0}},
        "preference": {"inc": {"ece": 0.2, "calibration": [{"count": 2}]}},
    }
    shard(
        tmp_path / "checkpoints",
        0,
        {"losses.json": [step], "diagnostics.json": [window]},
    )
    rows = list(iter_saved_runs(tmp_path, None))
    assert [row["group"] for row in rows] == [
        "saved-pretrain-steps",
        "saved-pretrain-windows",
    ]
    assert [row["step"] for row in rows] == [7, 7]
    assert rows[0]["values"] == {
        "phi": 2.0,
        "pi": 3.0,
        "P": None,
        "loss": 37.0,
        "lr.phi.body": 0.002,
        "counts.pairs": 0,
        "gradients.global.norm": 4.0,
        "outputs.P_inc": None,
        "outputs.pi_action.mean_confidence": 0.7,
        "components.P": None,
    }
    assert rows[1]["values"] == {
        "losses.phi": 93.0,
        "losses.P": None,
        "coverage.losses.P": 0,
        "preference.inc.ece": 0.2,
    }
    assert rows[0]["source"] == rows[1]["source"] == str(tmp_path)
    assert rows[0]["x_label"] == "Optimizer step"
    assert rows[0]["source_file"] == str(tmp_path / "checkpoints/000000/losses.json")
    assert rows[1]["source_file"] == str(
        tmp_path / "checkpoints/000000/diagnostics.json"
    )


def test_reference_keeps_condition_split_and_initial_step(tmp_path):
    conditions = {
        "joint": {
            "train": {"losses": {"P": 8.0}},
            "validation": {"losses": {"P": 9.0}},
        },
        "P_only": {"train": {"losses": {"P": 4.0}}},
    }
    shard(
        tmp_path / "reference",
        0,
        {"metrics.json": {"step": 0, "conditions": conditions}},
    )
    shard(
        tmp_path / "reference",
        1,
        {
            "metrics.json": {
                "step": 120,
                "conditions": {"joint": {"train": {"losses": {"P": 2.0}}}},
            }
        },
    )
    rows = list(iter_saved_runs(tmp_path, None))
    assert [row["step"] for row in rows] == [0, 0, 0, 120]
    assert [row["values"]["losses.P"] for row in rows] == [8.0, 9.0, 4.0, 2.0]
    assert len({row["source"] for row in rows}) == 3
    assert rows[0]["source"] == rows[3]["source"]
    assert rows[0]["source_file"] == str(tmp_path / "reference/000000/metrics.json")
    assert all(row["group"] == "saved-pretrain-reference" for row in rows)


def test_joint_retrain_and_rl_cycles_have_distinct_sources(tmp_path):
    pretrain, rl = tmp_path / "pretrain", tmp_path / "rl"
    phases = [
        pretrain,
        pretrain / "retrain",
        rl / "000000",
        rl / "000000/retrain",
        rl / "000001",
    ]
    for value, root in enumerate(phases):
        shard(
            root / "checkpoints",
            0,
            {"losses.json": [{"step": 1, "loss": value, "ppo": None}]},
        )
    rows = list(iter_saved_runs(pretrain, rl))
    assert len(rows) == len({row["source"] for row in rows}) == 5
    assert [row["values"]["loss"] for row in rows] == [0, 1, 2, 3, 4]
    assert all(row["step"] == 1 for row in rows)
    assert [row["group"] for row in rows] == ["saved-pretrain-steps"] * 2 + [
        "saved-rl-steps"
    ] * 3
    assert "RL cycle 1" in rows[-1]["title"]


def test_partial_shards_are_ignored_and_no_checkpoint_weights_needed(tmp_path):
    shard(tmp_path / "checkpoints", 0, {"losses.json": [{"step": 2, "loss": 5.0}]})
    pending = tmp_path / "checkpoints/000001.part"
    pending.mkdir()
    (pending / "losses.json").write_text("not complete JSON")
    (tmp_path / "checkpoints/000000/checkpoint.pt").write_bytes(b"not a torch file")
    assert [row["step"] for row in iter_saved_runs(tmp_path, None)] == [2]


@pytest.mark.parametrize("directory", ["checkpoints", "reference"])
def test_complete_shard_gaps_fail(tmp_path, directory):
    (tmp_path / directory / "000001").mkdir(parents=True)
    with pytest.raises(ValueError, match="000000 missing"):
        list(iter_saved_runs(tmp_path, None))


@pytest.mark.parametrize("kind", ["pretrain", "rl"])
def test_missing_explicit_directory_fails(tmp_path, kind):
    missing = tmp_path / "missing"
    roots = (missing, None) if kind == "pretrain" else (None, missing)
    with pytest.raises(FileNotFoundError):
        list(iter_saved_runs(*roots))


def test_empty_root_reports_no_records(tmp_path, capsys):
    assert list(iter_saved_runs(tmp_path, None)) == []
    assert capsys.readouterr().err == "saved training logs: no complete records\n"


def test_no_requested_saved_roots_are_silent(capsys):
    assert list(iter_saved_runs(None, None)) == []
    assert capsys.readouterr().err == ""
