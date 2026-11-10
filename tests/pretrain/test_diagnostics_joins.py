"""Block 33 exact ordinary/reference joins and strict top-level payloads."""

import json

import pytest
from preferences.test_diagnostics import patch_alias
from progress_bars import Bar, screen

from mcpref.pretrain.checkpoints import write_checkpoint

from .loop_run import Loop
from .loop_world import WORLD, World


def diagnostics(loop):
    return loop.repo / "w/diagnostics"


def records(loop):
    paths = sorted(diagnostics(loop).glob("pretrain/*/metrics.json"))
    return [json.loads(path.read_text()) for path in paths]


def small_loop(tmp_path, **changes):
    options = {
        "diagnostic_window": 2,
        "checkpoint_steps": 2,
        "reference_steps": 60,
        "epochs": 1,
        "batch": 64,
        "micro_batch": 64,
        "world": World(train=WORLD.train[:2], validation=WORLD.validation),
    }
    return Loop(tmp_path, **(options | changes)).write()


def test_ordinary_records_join_exact_step_and_unique_window_only(tmp_path):
    loop = small_loop(tmp_path)
    result = loop.run()
    assert result.error is None
    losses = [
        entry
        for checkpoint in sorted((loop.out / "checkpoints").iterdir())
        for entry in json.loads((checkpoint / "losses.json").read_text())
    ]
    windows = [
        entry
        for checkpoint in sorted((loop.out / "checkpoints").iterdir())
        for entry in json.loads((checkpoint / "diagnostics.json").read_text())
    ]
    ordinary = sorted(
        (record for record in records(loop) if record["id"].startswith("step:")),
        key=lambda record: record["context"]["steps"][0],
    )
    assert len(ordinary) == len(losses) == 2
    for record in ordinary:
        step = record["context"]["steps"][0]
        assert record["id"] == f"step:{step}"
        step_matches = [entry for entry in losses if entry["step"] == step]
        window_matches = [entry for entry in windows if entry["last_step"] == step]
        assert len(step_matches) == 1 and len(window_matches) <= 1
        assert record["joined"]["outputs"] == step_matches[0]["outputs"]
        if window_matches:
            assert (
                record["joined"]["window_preference"] == window_matches[0]["preference"]
            )
            assert record["joined"]["reason"] is None
        else:
            assert record["joined"]["window_preference"] is None
            assert record["joined"]["reason"] == "window_open"
        assert set(record["joined"]) == {
            "outputs",
            "window_preference",
            "reason",
            *(("policy",) if step == 2 else ()),
        }
        assert "outputs" not in record["metrics"]
        assert "window_preference" not in record["metrics"]
        assert "condition" not in record["metrics"]


def test_initial_reference_records_map_exact_condition_split_and_pairs(tmp_path):
    loop = small_loop(tmp_path)
    loop.run()
    source_dir = loop.out / "reference/000000"
    source = json.loads((source_dir / "metrics.json").read_text())
    assert (source_dir / "pairs.json").is_file()
    reference = [
        record for record in records(loop) if record["id"].startswith("reference:")
    ]
    assert [record["id"] for record in reference] == [
        f"reference:0:{split}:{condition}"
        for split in ("train", "validation")
        for condition in ("joint", "P_only", "phi_only", "initial")
    ]
    for record in reference:
        _, _, split, condition = record["id"].split(":")
        assert record["joined"]["condition"] == source["conditions"][condition][split]
        assert record["context"]["steps"] == [0]
        assert record["context"]["artifact_paths"] == [
            "w/pretrain/reference/000000/metrics.json",
            "w/pretrain/reference/000000/pairs.json",
        ]
        assert "condition" not in record["metrics"]


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_ordinary_join_rejects_zero_or_multiple_exact_steps(tmp_path, problem):
    loop = small_loop(tmp_path, diagnostic_window=1, checkpoint_steps=1)

    def corrupt(mp):
        def once(*args, _real=write_checkpoint, **kwargs):
            result = _real(*args, **kwargs)
            path = loop.checkpoint_dir(0) / "losses.json"
            entries = json.loads(path.read_text())
            if problem == "missing":
                entries.clear()
            else:
                entries.append(entries[0])
            path.write_text(json.dumps(entries, indent=2) + "\n")
            return result

        patch_alias(mp, write_checkpoint, "write_checkpoint", once)

    result = loop.run(patch=corrupt, raises=RuntimeError)
    assert result.error is not None
    assert not [record for record in records(loop) if record["id"] == "step:1"]


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_window_join_rejects_zero_or_multiple_exact_last_steps(tmp_path, problem):
    loop = small_loop(tmp_path)

    def corrupt(mp):
        def once(*args, _real=write_checkpoint, **kwargs):
            result = _real(*args, **kwargs)
            path = loop.checkpoint_dir(0) / "diagnostics.json"
            entries = json.loads(path.read_text())
            if problem == "missing":
                entries.clear()
            else:
                entries.append(entries[0])
            path.write_text(json.dumps(entries, indent=2) + "\n")
            return result

        patch_alias(mp, write_checkpoint, "write_checkpoint", once)

    result = loop.run(patch=corrupt, raises=RuntimeError)
    assert result.error is not None
    assert not [record for record in records(loop) if record["id"] == "step:2"]


def test_pretrain_progress_counts_reference_as_one_work_unit(tmp_path):
    loop = small_loop(tmp_path)
    first = loop.run()
    assert Bar("diagnostics pretrain", 3, 3, "step") in screen(first.err)
    (loop.out / "model.pt").unlink()
    second = loop.run()
    assert "diagnostics pretrain: already done" in screen(second.err)
