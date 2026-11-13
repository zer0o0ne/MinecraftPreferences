"""Independent plot oracles for population boundaries, missing values and snapshots."""

import json
import math

import pytest
from PIL import Image

from mcpref.diagnostics.charts import line_figure, save_png
from mcpref.diagnostics.plot import collect, main, registries, saved_points
from mcpref.diagnostics.series import scalar, source
from mcpref.infrastructure.shards.store import write_shard


def record(producer, step, value, *, source="pretrain", kind="pretrain", index=0):
    return {
        "version": 1,
        "kind": kind,
        "index": index,
        "id": str(index),
        "context": {
            "producer": producer,
            "source": source,
            "model": "model",
            "split": "train",
            "condition": "joint",
            "steps": [step],
        },
        "populations": {},
        "metrics": {"P": {"mean": value}},
        "joined": {},
    }


def loss_lines(records):
    entries = [
        entry
        for entry in registries(records)
        if entry["group"] == "losses" and "__total__" in entry["file"]
    ]
    points = collect(records, entries)
    return [points[entry["file"]]["P"] for entry in entries]


def test_training_and_reference_never_join_one_population():
    assert loss_lines(
        [record("pretrain", 100, 10), record("pretrain-reference", 100, 1)]
    ) == [[(100, 10)], [(100, 1)]]


def test_rl_local_steps_remain_in_separate_cycles():
    records = [
        record(f"update:{cycle}", step, value, source=f"rl/{cycle}", kind="rl")
        for cycle, step, value in [(0, 1, 10), (0, 2, 20), (1, 1, 100), (1, 2, 200)]
    ]
    assert loss_lines(records) == [[(1, 10), (2, 20)], [(1, 100), (2, 200)]]


def test_legacy_reference_uses_saved_loss_when_rows_are_unavailable():
    item = record("pretrain-reference", 100, None)
    item["joined"] = {"condition": {"losses": {"P": 0.75}}}
    assert scalar(source(item, "P")) == 0.75


def test_large_metric_groups_are_paginated_without_lost_series():
    item = record("update:0", 1, None, kind="rl")
    metrics = {f"update.group.{i}": {"mean": i} for i in range(13)}
    item["metrics"] = metrics
    entries = [entry for entry in registries([item]) if "__groups__" in entry["file"]]
    assert len(entries) == 3
    assert all(len(entry["series"]) <= 6 for entry in entries)
    assert {name for entry in entries for name in entry["series"]} == set(metrics)


def test_pretrain_nested_reliability_uses_latest_snapshot():
    records = [
        record("pretrain", step, 1, index=i) for i, step in enumerate((100, 200))
    ]
    for current, accuracy in zip(records, ([0.1, 0.9], [0.2, 0.8])):
        current["joined"] = {
            "policy": {
                "reliability": [
                    {"bin": i, "confidence": i * 0.5, "accuracy": value, "n": 4}
                    for i, value in enumerate(accuracy)
                ]
            }
        }
    entries = [
        entry for entry in registries(records) if "__reliability__" in entry["file"]
    ]
    assert len(entries) == 1
    points = collect(records, entries)
    assert points[entries[0]["file"]]["accuracy"] == [(0, 0.2), (1, 0.8)]
    assert entries[0]["snapshot"] == "step [200], record 1"
    later = record("pretrain", 201, 1, index=2)
    points = collect([*records, later], entries)
    assert points[entries[0]["file"]]["accuracy"] == [(0, 0.2), (1, 0.8)]
    assert entries[0]["snapshot"] == "step [200], record 1"
    later["joined"] = {"policy": {"reliability": []}}
    points = collect([*records, later], entries)
    assert points[entries[0]["file"]]["accuracy"] == []
    assert entries[0]["snapshot"] == "step [201], record 2"


def test_saved_series_break_at_null_and_absent_values():
    rows = [
        {
            "group": "saved-pretrain-steps",
            "source": "run",
            "title": "Pretrain",
            "step": step,
            "values": values,
        }
        for step, values in enumerate(({"P": 2}, {}, {"P": None}, {"P": 1}), 1)
    ]
    entries, points = saved_points(rows)
    values = points[entries[0]["file"]]["P"]
    assert values[0] == (1, 2) and values[3] == (4, 1)
    assert math.isnan(values[1][1]) and math.isnan(values[2][1])


def test_all_null_saved_records_remain_visible():
    entries, points = saved_points(
        [
            {
                "group": "saved-pretrain-steps",
                "source": "run",
                "title": "Pretrain",
                "step": 1,
                "values": {"P": None},
            }
        ]
    )
    assert len(entries) == 1
    assert math.isnan(points[entries[0]["file"]]["P"][0][1])


def test_separate_axes_visible_points_missing_gaps_and_atomic_png(tmp_path):
    result = line_figure(
        "Synthetic verification",
        {"loss": [(1, 2), (2, math.nan), (3, 1)], "learning rate": [(1, 0.0001)]},
        "Optimizer step",
        step=False,
    )
    assert len(result.axes) == 2
    assert result.axes[1].lines[0].get_marker() == "."
    assert math.isnan(result.axes[0].lines[0].get_ydata()[1])
    path = tmp_path / "plot.png"
    save_png(result, path)
    first = path.read_bytes()
    image = Image.open(path)
    assert image.format == "PNG" and image.width >= 1000
    assert not list(tmp_path.glob("*.part"))
    again = line_figure(
        "Synthetic verification",
        {"loss": [(1, 2), (2, math.nan), (3, 1)], "learning rate": [(1, 0.0001)]},
        "Optimizer step",
        step=False,
    )
    save_png(again, path)
    assert path.read_bytes() == first


def test_saved_run_cli_creates_gallery_without_diagnostic_records(tmp_path):
    run, report = tmp_path / "pretrain", tmp_path / "report"
    with write_shard(run / "checkpoints", 0) as staged:
        (staged / "losses.json").write_text(
            json.dumps(
                [
                    {"step": 1, "phi": 8.0, "pi": 2.0, "P": None, "loss": 10.0},
                    {"step": 2, "phi": 6.0, "pi": 1.0, "P": None, "loss": 7.0},
                ]
            )
        )
    before = (run / "checkpoints/000000/losses.json").read_bytes()
    assert (
        main([str(tmp_path / "absent"), str(report), "--pretrain-dir", str(run)]) == 0
    )
    files = list(report.glob("*.png"))
    assert len(files) == 12 and all(Image.open(path).format == "PNG" for path in files)
    page = (report / "index.html").read_text()
    assert files[0].name in page and str(run) in page
    assert (run / "checkpoints/000000/losses.json").read_bytes() == before
    assert not (tmp_path / "absent").exists()


def test_empty_directory_and_missing_input_are_explicit(tmp_path):
    diagnostics = tmp_path / "diag"
    diagnostics.mkdir()
    assert main([str(diagnostics), str(tmp_path / "plots")]) == 0
    assert "No completed observations" in (tmp_path / "plots/index.html").read_text()
    with pytest.raises(SystemExit):
        main([str(tmp_path / "typo"), str(tmp_path / "plots")])
