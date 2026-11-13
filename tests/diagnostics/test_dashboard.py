"""Dashboard contracts: twelve files, compatible overlays and honest missing data."""

import math

from diagnostics.test_plot_sources import record
from mcpref.diagnostics.plot import saved_points
from mcpref.diagnostics.report.build import dashboards
from mcpref.diagnostics.report.render import dashboard_figure
from mcpref.diagnostics.series import collect, registries


def saved(values, *, source="run", title="Pretrain / joint / steps"):
    return {
        "group": "saved-pretrain-steps",
        "source": source,
        "title": title,
        "step": 1,
        "values": values,
    }


def build(rows):
    entries, points = saved_points(rows)
    return dashboards(entries, points)


def test_twelve_stable_names_and_no_empty_axes():
    entries, panels = build([saved({"phi": 3, "pi": 2, "P": None, "loss": 5})])
    assert len(entries) == 12
    assert [entry["file"][:3] for entry in entries] == [
        f"{i:02d}_" for i in range(1, 13)
    ]
    assert all("__" not in entry["file"] for entry in entries)
    losses = panels["01_losses.png"]
    assert len(losses) == 1
    assert {curve["metric"] for curve in losses[0]["curves"]} == {"phi", "pi", "loss"}
    assert entries[0]["missing"] == 1
    figure = dashboard_figure(entries[0], losses)
    assert len(figure.axes) == 1 and len(figure.axes[0].lines) == 3
    figure.clear()


def test_learning_rates_and_norms_never_share_axis():
    _, panels = build(
        [
            saved(
                {
                    "lr.phi": 0.001,
                    "lr.pi": 0.002,
                    "gradients.groups.phi.norm": 500,
                    "gradients.groups.pi.norm": 100,
                    "updates.groups.phi.norm": 0.4,
                    "updates.groups.phi.relative": 0.01,
                }
            )
        ]
    )
    health = panels["05_optimization.png"]
    assert len(health) == 4
    lr = next(panel for panel in health if panel["unit"] == "Learning rate")
    assert len(lr["curves"]) == 2
    assert all("lr." in curve["metric"] for curve in lr["curves"])


def test_nulls_break_curves_without_drawing_missing_series():
    rows = [saved({"P": 2, "pi": None}), saved({"pi": None}), saved({"P": 1})]
    for step, row in enumerate(rows, 1):
        row["step"] = step
    _, panels = build(rows)
    curve = panels["01_losses.png"][0]["curves"][0]
    assert curve["points"][0] == (1, 2)
    assert math.isnan(curve["points"][1][1])
    assert curve["points"][2] == (3, 1)


def test_cycles_overlay_distinct_local_curves_and_retrain_keeps_own_axis():
    rows = [
        saved({"loss": value}, source=source, title=title)
        for value, source, title in (
            (1, "rl/000000", "RL cycle 0 / joint / steps"),
            (2, "rl/000001", "RL cycle 1 / joint / steps"),
            (3, "rl/000001/retrain", "RL cycle 1 / retrain / steps"),
        )
    ]
    _, panels = build(rows)
    assert len(panels["01_losses.png"]) == 2
    assert {
        curve["points"][0]
        for panel in panels["01_losses.png"]
        for curve in panel["curves"]
    } == {
        (1, 1),
        (1, 2),
        (1, 3),
    }


def test_empty_group_explains_required_source_instead_of_empty_axes():
    entries, panels = build([saved({"P": None})])
    entry = entries[6]
    assert entry["file"] == "07_ppo.png"
    figure = dashboard_figure(entry, panels[entry["file"]])
    assert not figure.axes
    assert any("RL" in text.get_text() for text in figure.texts)
    figure.clear()


def test_open_windows_do_not_insert_fake_missing_measurements():
    records = [record("pretrain", step, 0, index=step) for step in (1, 2, 3)]
    for row, value in zip(records, (0.1, None, 0.3)):
        row["joined"]["window_preference"] = (
            None if value is None else {"inc": {"brier": value}}
        )
    entries = registries(records)
    _, panels = dashboards(entries, collect(records, entries))
    curves = [
        curve
        for panel in panels["02_preferences.png"]
        for curve in panel["curves"]
        if curve["metric"] == "preference.inc.brier"
    ]
    assert len(curves) == 1
    assert curves[0]["points"] == [(1, 0.1), (3, 0.3)]
    records[1]["joined"]["window_preference"] = {"inc": {"brier": None}}
    _, panels = dashboards(entries, collect(records, entries))
    curve = next(
        c
        for p in panels["02_preferences.png"]
        for c in p["curves"]
        if c["metric"] == "preference.inc.brier"
    )
    assert math.isnan(curve["points"][1][1])


def test_additional_saved_producer_metrics_and_real_action_axis():
    item = record("ppo:0", 5, None, kind="rl")
    item["metrics"] = {
        "ppo.eligible_count": 17,
        "ppo.kl_exact_undefined": 2,
        "ppo.kl_exact": {"mean": 0.1},
    }
    mcts = record("mcts:0", 500, None, kind="mcts")
    mcts["metrics"] = {"mask.count": {"mean": 6}}
    entries = registries([item, mcts])
    _, panels = dashboards(entries, collect([item, mcts], entries))
    ppo = panels["07_ppo.png"]
    assert {c["metric"] for p in ppo for c in p["curves"]} == set(item["metrics"])
    assert len(ppo) == 2
    assert panels["03_policy.png"][0]["x_label"] == "Real action"


def test_gallery_migrates_only_previously_owned_pngs(tmp_path):
    from mcpref.diagnostics.gallery import write_gallery

    old = "saved-pretrain-steps__abcdef012345__0.png"
    (tmp_path / old).write_bytes(b"old generated PNG")
    (tmp_path / "personal.png").write_bytes(b"user file")
    (tmp_path / "index.html").write_text(
        f'<h1>Training diagnostics</h1><img src="{old}"><img src="personal.png">'
    )
    entries, _ = build([])
    write_gallery(tmp_path, entries)
    assert not (tmp_path / old).exists()
    assert (tmp_path / "personal.png").read_bytes() == b"user file"
    assert "01_losses.png" in (tmp_path / "index.html").read_text()


def test_memory_cycle_axis_preserves_saved_counts_and_run_boundaries():
    rows = []
    for run, cycle, count in (("a", 0, 3), ("a", 1, 7), ("b", 0, 99)):
        item = record(
            f"memory:{cycle:06d}",
            cycle,
            None,
            source=f"{run}/{cycle:06d}/run.json",
            kind="rl",
        )
        item["metrics"] = {"memory.seeds": count}
        rows.append(item)
    entries = registries(rows)
    _, panels = dashboards(entries, collect(rows, entries))
    assert len(panels["11_memory.png"]) == 2
    assert panels["11_memory.png"][0]["curves"][0]["points"] == [(0, 3), (1, 7)]
    assert panels["11_memory.png"][1]["curves"][0]["points"] == [(0, 99)]
    assert all(panel["x_label"] == "Cycle" for panel in panels["11_memory.png"])


def test_reference_overlays_keep_different_conditions_and_splits():
    rows = []
    for split, condition, value in (
        ("train", "joint", 1),
        ("train", "P_only", 2),
        ("validation", "joint", 3),
    ):
        item = record("pretrain-reference", 500, value)
        item["context"].update(split=split, condition=condition)
        rows.append(item)
    entries = registries(rows)
    _, panels = dashboards(entries, collect(rows, entries))
    losses = next(
        panel
        for panel in panels["06_reference.png"]
        if panel["title"] == "Training objectives"
    )
    assert len(losses["curves"]) == 3
    assert {c["label"] for c in losses["curves"]} == {
        "P · train · joint",
        "P · train · P_only",
        "P · validation · joint",
    }
    assert {c["points"][0] for c in losses["curves"]} == {(500, 1), (500, 2), (500, 3)}
