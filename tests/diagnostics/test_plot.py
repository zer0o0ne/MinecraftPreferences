"""Block 33 plot registry, exact facets, joined aliases and cheap replot."""

import copy
import gc
import json
import weakref

import numpy as np

from mcpref.diagnostics.artifacts import write_record
from mcpref.diagnostics.plot import main
from mcpref.diagnostics.targets import on_graph, on_label_shard, on_vote_shard
from mcpref.pretrain.diagnostics import on_pretrain_step, on_reference

from .world import comparison, config, context, fit, label, pair, rows, vote

TARGET_FILES = {
    "target-provenance__raw-vote-0.svg",
    "target-provenance__raw-vote-1.svg",
    "target-provenance__graph-removed.svg",
    "target-provenance__all-pairs.svg",
    "target-provenance__aggregate.svg",
    "target-provenance__model.svg",
}
PANELS = (
    "losses",
    "progress",
    "brier-inc",
    "brier-succ",
    "calibration-inc",
    "calibration-succ",
)


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


def test_registry_full_actual_paths_every_facet_and_literal_dotted_lookup(tmp_path):
    out, plots = tmp_path / "diag", tmp_path / "plots"
    records = make_records()
    write_records(out, zip(("targets", "pretrain"), records))
    before = copy.deepcopy(records)
    assert main([str(out), str(plots)]) == 0
    expected = set(TARGET_FILES)
    for split in ("train", "validation"):
        for condition in ("joint", "P_only", "phi_only"):
            for panel in PANELS:
                expected.add(f"pretrain-preference__{split}__{condition}__{panel}.svg")
            for panel in ("total", "components"):
                expected.add(f"losses__{split}__{condition}__{panel}.svg")
    actual = {path.name for path in plots.rglob("*.svg")}
    assert actual == expected
    assert records == before
    index = json.loads((out / "index.json").read_text())
    assert {entry["file"] for entry in index["plots"]} == expected
    assert all(
        entry["file"].startswith(entry["group"] + "__") for entry in index["plots"]
    )
    assert all((plots / entry["file"]).is_file() for entry in index["plots"])
    model = next(
        entry for entry in index["plots"] if entry["file"].endswith("__model.svg")
    )
    assert model["series"] == ["target.model_lt_0_2", "target.model_gt_0_8"]
    brier = (plots / "pretrain-preference__train__joint__brier-inc.svg").read_text()
    assert "0.125" in brier and "99" not in brier
    for name in ("losses.svg", "progress.svg", "brier-inc.svg"):
        path = plots / f"pretrain-preference__train__joint__{name}"
        assert 'id="axes_2"' not in path.read_text()
    assert not list(plots.rglob("*.part"))


def test_replot_overwrites_deterministically_without_recomputation(
    tmp_path, monkeypatch
):
    out, plots = tmp_path / "diag", tmp_path / "plots"
    write_records(out, zip(("targets", "pretrain"), make_records()))
    main([str(out), str(plots)])
    before = {path: path.read_bytes() for path in plots.rglob("*.svg")}
    victim = plots / "target-provenance__model.svg"
    victim.write_text("stale")

    def forbidden(*args, **kwargs):
        raise AssertionError("plot recomputed bootstrap draws")

    monkeypatch.setattr(np.random, "Generator", forbidden)
    assert main([str(out), str(plots)]) == 0
    after = {path: path.read_bytes() for path in plots.rglob("*.svg")}
    assert after == before and not list(plots.rglob("*.part"))


def test_plot_streams_records_and_draws_group_progress(tmp_path, monkeypatch, capsys):
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
    from progress_bars import Bar, screen, tqdm_defaults

    tqdm_defaults(monkeypatch)
    assert main([str(out), str(plots)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert screen(captured.err) == [Bar("diagnostics plot", 13, 13, "group")]
