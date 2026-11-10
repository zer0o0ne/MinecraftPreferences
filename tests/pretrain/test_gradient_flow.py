"""Selected-step, counterfactual, summary and accepted-artifact oracles of specs/pretrain-diagnostics.md (22)-(29)."""

import copy
import json
import math

import numpy as np
import pytest
import torch
from training.schedule_oracle import epoch

from mcpref.pretrain.gradient_flow import (
    GradientStepRecord,
    baseline_pairs,
    post_step_pairs,
    summarize,
)

from .gradient_oracle import select
from .loop_modules import fakes
from .loop_run import Loop
from .loop_world import NO_PAIRS, World, clip, p, u
from .test_reference_resume import nested_equal, optimizer_of, snapshot


def selection_oracle(world: World, cfg: dict) -> dict:
    rng = np.random.Generator(np.random.PCG64([cfg["seed"], 3]))
    sizes = [len(shard) for shard in world.train]
    entries = [
        entry for _ in range(cfg["epochs"]) for entry in epoch(rng, sizes, cfg["batch"])
    ]
    labelled = [t for t, (shard, positions) in enumerate(entries, 1)
                if any("clip" not in world.train[shard][i] for i in positions)]  # fmt: skip
    target = max(1, len(entries) // 2)
    beginning = labelled[0] if labelled else None
    middle = next(
        (t for t in labelled if t >= target), labelled[-1] if labelled else None
    )
    roles = {}
    for step, role in (
        (beginning, "beginning"),
        (middle, "middle"),
        (labelled[-1] if labelled else None, "end"),
    ):
        if step is not None:
            roles.setdefault(step, []).append(role)
    return {"version": 1, "middle_target": target,
            "steps": [{"step": step, "roles": names} for step, names in sorted(roles.items())]}  # fmt: skip


def test_selected_replay_matches_schedule_and_preserves_numpy_rng(
    tmp_path, monkeypatch, capsys
):
    from progress_bars import Bar, screen, tqdm_defaults

    tqdm_defaults(monkeypatch)
    loop = Loop(tmp_path).write()
    rng = np.random.Generator(np.random.PCG64([5, 3]))
    before = rng.bit_generator.state
    got = select(loop)
    assert got == selection_oracle(loop.world, loop.config)
    assert got["steps"] == [
        {"step": 1, "roles": ["beginning"]},
        {"step": 5, "roles": ["middle"]},
        {"step": 9, "roles": ["end"]},
    ]
    assert rng.bit_generator.state == before
    assert screen(capsys.readouterr().err)[-1] == Bar(
        "pretrain gradient flow selection", 10, 10, "step"
    )


@pytest.mark.parametrize(
    "world, expected",
    [
        (World(train=[[u(0), p(0), u(1)], [p(1)]]), [1, 4]),
        (World(train=[[p(0), u(0), u(1)], [u(2)]]), [2]),
        (World(train=[[p(0)]]), [1]),
    ],
    ids=["gaps", "fallback and deduplication", "tiny"],
)
def test_selected_replay_gaps_fallback_and_deduplicated_roles(
    tmp_path, world, expected
):
    loop = Loop(tmp_path, world=world, epochs=1, batch=1).write()
    got = select(loop)
    assert got == selection_oracle(world, loop.config)
    assert [x["step"] for x in got["steps"]] == expected
    assert all(
        x["roles"]
        == [role for role in ("beginning", "middle", "end") if role in x["roles"]]
        for x in got["steps"]
    )


def test_selected_replay_without_labelled_steps_has_no_work(tmp_path, capsys):
    loop = Loop(tmp_path, world=NO_PAIRS).write()
    got = select(loop)
    assert got["steps"] == []
    assert "pretrain gradient flow: nothing to do" in capsys.readouterr().err
    assert not (loop.out / "gradient-flow/work").exists()


def pair_loss(mods: dict, item: dict) -> float:
    z = mods["phi"]([clip(item["a"]), clip(item["b"])])
    out = mods["P"](z[0:1], z[1:2])
    return float(mods["P_loss"].terms(out, [item["labels"]])["total"][0].detach())


def test_counterfactual_raw_pairs_deltas_calls_modes_and_state(tmp_path, monkeypatch):
    from mcpref.pretrain import gradient_pairs
    from mcpref.pretrain.checks import index_run
    from mcpref.pretrain.config import load_config

    from .index_run import repo_root

    loop = Loop(tmp_path).write()
    mods = fakes(noise=False, stateful=False)
    items = [p(0), p(1)]
    contexts = []
    real = gradient_pairs.evaluation_state

    def counted(*args, **kwargs):
        contexts.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(gradient_pairs, "evaluation_state", counted)
    oracle_before = copy.deepcopy(mods)
    expected_baseline = [pair_loss(oracle_before, item) for item in items[:2]]
    with repo_root(loop.repo):
        cfg = load_config(loop.path)
        run = index_run(cfg)
        baseline = baseline_pairs(cfg, mods, items, run)
    assert baseline == pytest.approx(expected_baseline)
    frozen = copy.deepcopy(mods["P"])
    frozen.eval()
    with torch.no_grad():
        mods["phi"].A.add_(0.125)
        mods["P"].u.add_(0.25)
        mods["P"].v.sub_(0.5)
    optimizer = optimizer_of(mods)
    before = snapshot(mods, optimizer)
    with repo_root(loop.repo):
        only_phi, joint = post_step_pairs(cfg, mods, frozen, items, run)
    assert nested_equal(snapshot(mods, optimizer), before)
    assert len(contexts) == 2 and frozen.training is False
    oracle = copy.deepcopy(mods)
    expected_only = [pair_loss(oracle, item) for item in items[:2]]
    oracle["P"] = frozen
    expected_frozen = [pair_loss(oracle, item) for item in items[:2]]
    assert only_phi == pytest.approx(expected_frozen)
    assert joint == pytest.approx(expected_only)
    assert [x - y for x, y in zip(only_phi, baseline)] == pytest.approx(
        [x - y for x, y in zip(expected_frozen, expected_baseline)]
    )
    assert [x - y for x, y in zip(joint, baseline)] == pytest.approx(
        [x - y for x, y in zip(expected_only, expected_baseline)]
    )
    assert mods["phi"].calls == 4 and mods["P"].calls == 4


def bundle(norms=(1.0, 2.0, 3.0), dots=(-1.0, 2.0, 1.0, 4.0), update=-2.0) -> dict:
    return {"norms": dict(zip(("phi", "pi", "P"), norms)),
            "dot": dict(zip(("P_phi", "P_pi", "P_other", "P_joint"), dots)),
            "cosine": {"P_phi": -0.5, "P_pi": 0.5, "P_other": 0.25},
            "update_dot": update, "update_unavailable": None}  # fmt: skip


def accepted(
    step: int, delta_only: list[float], delta_joint: list[float], update=-2.0
) -> GradientStepRecord:
    checkpoint = step // 3
    directory = f"checkpoints/{checkpoint:06d}/gradient_flow/{checkpoint:06d}"
    pairs = [{"micro_batch": i, "pair_in_micro": 0, "a": "a", "b": "b", "labels": {},
              "baseline": 0.0, "only_phi_post": x, "joint_post": y,
              "delta_only_phi": x, "delta_joint": y} for i, (x, y) in enumerate(zip(delta_only, delta_joint))]  # fmt: skip
    metrics = {"version": 1, "step": step, "roles": ["beginning"], "checkpoint_losses_index": step % 3 - 1,
               "micro_batches": [], "phi": {"groups": {}, "total": bundle(update=update)},
               "P": {"groups": {}, "total": {"norm": 5.0 if step == 1 else None, "update_dot": 3.0, "update_unavailable": None}},
               "nonfinite": {}, "counterfactual": {}, "evidence": {"kind": "synthetic", "checkpoint_shard": f"checkpoints/{checkpoint:06d}", "pairs": f"{directory}/pairs.json"}}  # fmt: skip
    return GradientStepRecord(metrics, pairs)


def test_summary_excludes_synthetic_evidence():
    records = [
        accepted(1, [-1.0, 0.0], [1.0, -1.0], update=-2.0),
        accepted(4, [2.0, -2.0], [0.0, 2.0], update=4.0),
    ]
    got = summarize(records)
    assert got["steps"] == []
    assert all(
        x["count"] == 0 and x["median"] is None
        for x in got["step_distributions"].values()
    )
    assert all(
        x["count"] == 0 and x["mean"] is None
        for x in got["pair_distributions"].values()
    )


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )


def test_loop_summary_matches_independent_step_and_pair_distributions(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    records = []
    for j, selected in enumerate(
        json.loads((loop.out / "gradient-flow/selection.json").read_text())["steps"]
    ):
        directory = loop.checkpoint_dir(j) / "gradient_flow" / f"{j:06d}"
        metrics = json.loads((directory / "metrics.json").read_text())
        pairs = json.loads((directory / "pairs.json").read_text())
        records.append(GradientStepRecord(metrics, pairs))
    got = summarize(records)
    paths = {"phi.total.norms.phi": ("phi", "norms", "phi"), "phi.total.norms.pi": ("phi", "norms", "pi"),
             "phi.total.norms.P": ("phi", "norms", "P"), "phi.total.dot.P_phi": ("phi", "dot", "P_phi"),
             "phi.total.dot.P_pi": ("phi", "dot", "P_pi"), "phi.total.dot.P_other": ("phi", "dot", "P_other"),
             "phi.total.dot.P_joint": ("phi", "dot", "P_joint"), "phi.total.cosine.P_phi": ("phi", "cosine", "P_phi"),
             "phi.total.cosine.P_pi": ("phi", "cosine", "P_pi"), "phi.total.cosine.P_other": ("phi", "cosine", "P_other"),
             "phi.total.update_dot": ("phi", "update_dot", None), "P.total.norm": ("P", "norm", None),
             "P.total.update_dot": ("P", "update_dot", None),
             "counterfactual.delta_only_phi.mean": (None, "delta_only_phi", "mean"),
             "counterfactual.delta_joint.mean": (None, "delta_joint", "mean")}  # fmt: skip
    assert set(got["step_distributions"]) == set(paths)
    for path, (head, section, key) in paths.items():
        if head is None:
            values = [
                record.metrics["counterfactual"][section]["mean"] for record in records
            ]
        elif key is None:
            values = [record.metrics[head]["total"][section] for record in records]
        else:
            values = [record.metrics[head]["total"][section][key] for record in records]
        values = [x for x in values if x is not None]
        distribution = got["step_distributions"][path]
        assert distribution["count"] == len(values)
        assert distribution["median"] == (median(values) if values else None)
        assert distribution["fraction_negative"] == pytest.approx(
            sum(x < 0 for x in values) / len(values) if values else 0
        )
        assert distribution["fraction_positive"] == pytest.approx(
            sum(x > 0 for x in values) / len(values) if values else 0
        )
        assert distribution["fraction_zero"] == pytest.approx(
            sum(x == 0 for x in values) / len(values) if values else 0
        )
    for name in ("delta_only_phi", "delta_joint"):
        values = [pair[name] for record in records for pair in record.pairs]
        distribution = got["pair_distributions"][name]
        assert distribution["count"] == len(values)
        assert distribution["mean"] == pytest.approx(sum(values) / len(values))
        assert distribution["median"] == median(values)
        assert distribution["harmful"] == pytest.approx(
            sum(x > 0 for x in values) / len(values)
        )
        assert distribution["helpful"] == pytest.approx(
            sum(x < 0 for x in values) / len(values)
        )
        assert distribution["zero"] == pytest.approx(
            sum(x == 0 for x in values) / len(values)
        )
        assert set(distribution["quantiles"]) == {"0.05", "0.5", "0.95"}


def scheduled_pairs(world: World, cfg: dict, step: int) -> list[tuple]:
    rng = np.random.Generator(np.random.PCG64([cfg["seed"], 3]))
    sizes = [len(shard) for shard in world.train]
    entries = [
        entry for _ in range(cfg["epochs"]) for entry in epoch(rng, sizes, cfg["batch"])
    ]
    shard, positions = entries[step - 1]
    items = [world.train[shard][i] for i in positions]
    result = []
    for micro, start in enumerate(range(0, len(items), cfg["micro_batch"])):
        labelled = [
            item
            for item in items[start : start + cfg["micro_batch"]]
            if "clip" not in item
        ]
        for ordinal, item in enumerate(labelled):
            result.append((micro, ordinal, item["a"], item["b"], item["labels"]))
    return result


def test_loop_accepted_artifacts_schema_indices_evidence_and_summary(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    selection = json.loads((loop.out / "gradient-flow/selection.json").read_text())
    assert selection == selection_oracle(loop.world, loop.config)
    for index, selected in enumerate(selection["steps"]):
        directory = loop.checkpoint_dir(index) / "gradient_flow" / f"{index:06d}"
        metrics = json.loads((directory / "metrics.json").read_text())
        pairs = json.loads((directory / "pairs.json").read_text())
        step = selected["step"]
        assert metrics["step"] == step and metrics["roles"] == selected["roles"]
        losses = loop.losses(index)
        assert metrics["checkpoint_losses_index"] == next(
            i for i, entry in enumerate(losses) if entry["step"] == step
        )
        joined = losses[metrics["checkpoint_losses_index"]]
        assert joined["gradients"]["global"]["norm"] is not None and joined["lr"]
        assert metrics["evidence"] == {"kind": "real_pretrain_step", "checkpoint_shard": f"checkpoints/{index:06d}",
                                       "pairs": f"checkpoints/{index:06d}/gradient_flow/{index:06d}/pairs.json"}  # fmt: skip
        assert (loop.out / metrics["evidence"]["pairs"]).exists()
        assert set(metrics) == {"version", "step", "roles", "checkpoint_losses_index", "micro_batches", "phi", "P",
                                "nonfinite", "counterfactual", "evidence"}  # fmt: skip
        assert "lr" not in metrics and "clip_coefficient" not in metrics["phi"]["total"]
        for head in ("phi", "P"):
            bundles = [*metrics[head]["groups"].values(), metrics[head]["total"]]
            assert all("update_unavailable" in bundle for bundle in bundles)
            updates = [
                bundle["update_dot"]
                for bundle in metrics[head]["groups"].values()
                if bundle["update_dot"] is not None
            ]
            assert metrics[head]["total"]["update_dot"] == pytest.approx(sum(updates))
        assert metrics["P"]["total"]["norm"] == pytest.approx(
            math.sqrt(
                sum(bundle["norm"] ** 2 for bundle in metrics["P"]["groups"].values())
            )
        )
        assert [
            (
                pair["micro_batch"],
                pair["pair_in_micro"],
                pair["a"],
                pair["b"],
                pair["labels"],
            )
            for pair in pairs
        ] == scheduled_pairs(loop.world, loop.config, step)
        for pair in pairs:
            assert set(pair) == {"micro_batch", "pair_in_micro", "a", "b", "labels", "baseline", "only_phi_post",
                                 "joint_post", "delta_only_phi", "delta_joint"}  # fmt: skip
            assert pair["pair_in_micro"] == 0 and pair["micro_batch"] >= 0
            assert pair["delta_only_phi"] == pytest.approx(
                pair["only_phi_post"] - pair["baseline"]
            )
            assert pair["delta_joint"] == pytest.approx(
                pair["joint_post"] - pair["baseline"]
            )
        for name in ("delta_only_phi", "delta_joint"):
            values = [pair[name] for pair in pairs]
            stats = metrics["counterfactual"][name]
            assert stats["count"] == len(values) and stats["mean"] == pytest.approx(
                sum(values) / len(values)
            )
            assert stats["harmful"] == pytest.approx(
                sum(x > 0 for x in values) / len(values)
            )
            assert stats["helpful"] == pytest.approx(
                sum(x < 0 for x in values) / len(values)
            )
            assert stats["zero"] == pytest.approx(
                sum(x == 0 for x in values) / len(values)
            )
    summary = json.loads((loop.out / "gradient-flow/summary.json").read_text())
    assert (
        len(summary["steps"]) == 3
        and summary["pair_distributions"]["delta_only_phi"]["count"] == 3
    )
    for row in summary["steps"]:
        assert (loop.out / row["metrics"]).exists()
        assert (
            row["delta_only_phi_mean"] is not None
            and row["delta_joint_mean"] is not None
        )
    assert not (loop.out / "gradient-flow/work").exists()


def test_training_results_bitwise_identical_without_selected_instrumentation(tmp_path):
    import mcpref.pretrain.loop as loop_module
    from mcpref.pretrain import gradient_flow

    whole = Loop(tmp_path / "whole", keep_checkpoints=20).write()
    whole.run()
    target = (
        loop_module if hasattr(loop_module, "select_gradient_steps") else gradient_flow
    )
    empty = {"version": 1, "middle_target": 5, "steps": []}
    other = Loop(tmp_path / "other", keep_checkpoints=20).write()
    other.run(
        patch=lambda mp: mp.setattr(
            target, "select_gradient_steps", lambda *args: empty
        )
    )
    for j in range(4):
        for name in ("losses.json", "diagnostics.json"):
            assert (whole.checkpoint_dir(j) / name).read_bytes() == (
                other.checkpoint_dir(j) / name
            ).read_bytes()
        assert nested_equal(whole.checkpoint(j), other.checkpoint(j))
    assert nested_equal(whole.model(), other.model())
    for name in ("selection.json", "baselines.json", "initial.pt", "metrics.json"):
        assert (whole.out / "reference/000000" / name).read_bytes() == (
            other.out / "reference/000000" / name
        ).read_bytes()


def test_staged_checkpoint_and_stale_work_are_not_accepted_on_resume(tmp_path):
    from mcpref.pretrain import steps

    from .index_run import Interrupted

    whole = Loop(tmp_path / "whole", keep_checkpoints=20).write()
    whole.run()
    cut = Loop(tmp_path / "cut", keep_checkpoints=20).write()
    target = steps

    def interrupt(*args, **kwargs):
        raise Interrupted("staged checkpoint")

    cut.run(
        patch=lambda mp: mp.setattr(target, "write_checkpoint", interrupt),
        raises=Interrupted,
    )
    stale = cut.out / "gradient-flow/work/999999"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "junk").write_text("stale")
    cut.run()
    for j in range(4):
        for name in ("losses.json", "diagnostics.json"):
            assert (cut.checkpoint_dir(j) / name).read_bytes() == (
                whole.checkpoint_dir(j) / name
            ).read_bytes()
    expected = {name for name in whole.tree() if "gradient_flow/" in name}
    assert {name for name in cut.tree() if "gradient_flow/" in name} == expected
    for name in expected:
        assert (cut.out / name).read_bytes() == (whole.out / name).read_bytes()
    assert not (cut.out / "gradient-flow/work").exists()


def test_gradient_progress_totals_and_selection_reuse(tmp_path):
    from progress_bars import assert_progress, screen, states

    loop = Loop(tmp_path).write()
    result = loop.run()
    assert_progress(result.err)
    assert states(result.err, "pretrain gradient flow selection")[-1].total == 10

    def totals(desc):
        values, previous = [], None
        for bar in states(result.err, desc):
            if bar.n == 0 and (previous is None or previous.n > 0):
                values.append(bar.total)
            previous = bar
        return values

    assert totals("pretrain gradient flow collect") == [2, 1, 2]
    assert totals("pretrain gradient flow reduce") == [20, 15, 20]
    assert totals("pretrain gradient flow pairs") == [2, 2, 2]
    (loop.out / "model.pt").unlink()
    mods = loop.modules()
    resumed = loop.run(mods)
    assert "pretrain gradient flow selection: already done" in screen(resumed.err)
    assert all(module.calls == 0 for module in mods.values())
