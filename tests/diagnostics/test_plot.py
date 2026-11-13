"""Block 33 plot registry, exact facets, joined aliases and cheap replot."""

import copy
import gc
import json
import weakref

import numpy as np

from diagnostics.core.world import (
    comparison,
    config,
    context,
    fit,
    label,
    pair,
    rows,
    vote,
)
from mcpref.diagnostics.core.artifacts import write_record
from mcpref.diagnostics.plot import main
from mcpref.diagnostics.pretrain.diagnostics import on_pretrain_step, on_reference
from mcpref.diagnostics.producers.targets import on_graph, on_label_shard, on_vote_shard


def preference():
    return {
        "inc": {"brier": 0.125, "ece": 0.25},
        "succ": {"brier": 0.375, "ece": 0.5},
        "inc.brier": 99.0,
    }


def make_records():
    targets = [
        on_vote_shard(
            [vote("e0", ["LOW"])],
            weights={"LOW": 0.1},
            cfg=config(),
            context=context("pair_votes/shards/000000/answers.json", "vlm-labels"),
        ),
        on_label_shard([label("e0", 0.9, 0)], cfg=config(), context=context()),
        on_graph(
            [comparison("e0")],
            categories=["comparable"],
            cfg=config(),
            context=context("pair_labels/shards/000003/counts.json", "acyclic-graph"),
        ),
    ]
    step = on_pretrain_step(
        rows(),
        fit=fit(),
        cfg=config(),
        context=context(
            "checkpoints/000004/losses.json", "pretrain", "joint", steps=(400,)
        ),
    )
    step["joined"] = {
        "outputs": {"P_inc": {"brier": 0.125}, "P_succ": {"brier": 0.375}},
        "window_preference": preference(),
    }
    pretrain = [step]
    for split in ("train", "validation"):
        for condition in ("joint", "P_only", "phi_only"):
            record = on_reference(
                rows(),
                pairs=[pair()],
                cfg=config(),
                context=context(
                    "reference/000001/metrics.json",
                    "pretrain-reference",
                    condition,
                    split=split,
                    steps=(500,),
                    clusters=("e0", "e1", "e2"),
                ),
            )
            record["joined"] = {
                "condition": {"preference": preference(), "losses": {"P": 1.0}}
            }
            pretrain.append(record)
    return targets, pretrain


def write_records(directory, records):
    for kind, group in records:
        for index, record in enumerate(group):
            write_record(directory, kind, index, record)


def fake_render(directory, entries, points):
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (directory / entry["file"]).write_bytes(b"routing-only fixture")


def test_registry_separates_populations_and_preserves_source_values(
    tmp_path, monkeypatch
):
    from mcpref.diagnostics import plot

    out, plots = tmp_path / "diag", tmp_path / "plots"
    records = make_records()
    write_records(out, zip(("targets", "pretrain"), records))
    before = copy.deepcopy(records)
    monkeypatch.setattr(plot, "render", fake_render)
    assert main([str(out), str(plots)]) == 0
    index = json.loads((out / "index.json").read_text())
    assert records == before
    assert all((plots / entry["file"]).is_file() for entry in index["plots"])
    assert all(entry["file"].endswith(".png") for entry in index["plots"])
    entries = plot.registries(plot.iter_all_records(out))
    points = plot.collect(plot.iter_all_records(out), entries)
    briers = [entry for entry in entries if "__brier-inc__" in entry["file"]]
    assert len(briers) == 7  # ordinary train plus six fixed-reference populations
    assert all(
        points[entry["file"]]["preference.inc.brier"]
        == [(400 if entry["identity"][1] == "pretrain" else 500, 0.125)]
        for entry in briers
    )
    assert "Training diagnostics" in (plots / "index.html").read_text()
    assert not list(plots.rglob("*.part"))


def test_replot_overwrites_deterministically_without_recomputation(
    tmp_path, monkeypatch
):
    from mcpref.diagnostics import plot

    out, plots = tmp_path / "diag", tmp_path / "plots"
    write_records(out, zip(("targets", "pretrain"), make_records()))
    monkeypatch.setattr(plot, "render", fake_render)
    main([str(out), str(plots)])
    before = {path: path.read_bytes() for path in plots.iterdir()}
    next(plots.glob("*.png")).write_bytes(b"stale")

    def forbidden(*args, **kwargs):
        raise AssertionError("plot recomputed bootstrap draws")

    monkeypatch.setattr(np.random, "Generator", forbidden)
    assert main([str(out), str(plots)]) == 0
    assert {path: path.read_bytes() for path in plots.iterdir()} == before


def test_plot_streams_records(tmp_path, monkeypatch):
    from mcpref.diagnostics import plot

    out, plots = tmp_path / "diag", tmp_path / "plots"
    write_records(out, [("targets", make_records()[0])])
    real_loads = json.loads
    previous = []

    class WeakRecord(dict):
        pass

    def loads(value, *args, **kwargs):
        result = real_loads(value, *args, **kwargs)
        if isinstance(result, dict) and "populations" in result and "joined" in result:
            result = WeakRecord(result)
            if len(previous) >= 2:
                gc.collect()
                assert previous[-2]() is None, "plot retained an older metric shard"
            previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(json, "loads", loads)
    monkeypatch.setattr(plot, "render", fake_render)
    assert main([str(out), str(plots)]) == 0
