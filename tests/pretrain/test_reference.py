"""Global-fit lifecycle and fixed-reference tests of specs/pretrain-diagnostics.md."""

import dataclasses
import json
import math

import pytest

from mcpref.pretrain.diagnostics import GlobalFit, Stratum
from mcpref.pretrain.reference import load_global_fit, write_global_fit

from .loop_run import Loop
from .loop_world import World, p
from .reference_oracle import (
    count_item_reads,
    full_world,
    labelled_world,
    select,
    selection_of,
    write_selection,
)


def fit(inc_probability=1 / 3, succ_probability=0.5) -> GlobalFit:
    strata = (Stratum("0.0", "comparable", 0.0, 1, 0),
              Stratum("1.0", "comparable", 1.0, 1, 0), Stratum("inc", "inc", None, 1, 1))  # fmt: skip
    logit = lambda x: (
        None if x is None or x in (0, 1) else __import__("math").log(x / (1 - x))
    )
    comparable = 0 if succ_probability is None else 2
    return GlobalFit(3, 3 - comparable, comparable, None if comparable == 0 else 1.0,
                     inc_probability, succ_probability, logit(inc_probability), logit(succ_probability), strata)  # fmt: skip


def test_loop_writes_hand_fit_and_complete_reference_without_staging(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    path = loop.out / "global-fit.json"
    assert not (loop.out / "reference/000000.part").exists()
    assert not path.with_suffix(".json.part").exists()
    assert (
        path.read_bytes()
        == (json.dumps(dataclasses.asdict(fit()), indent=2) + "\n").encode()
    )
    assert (
        loop.out / "reference/000000/baselines.json"
    ).read_bytes() == path.read_bytes()
    assert load_global_fit(loop.out) == fit()


@pytest.mark.parametrize(
    "labels, inc_probability, succ_probability, sum_p",
    [([0.25, 0.75], 0.0, 0.5, 1.0), ([None, None], 1.0, None, None)],
    ids=["all comparable", "all incomparable"],
)
def test_loop_degenerate_global_fits(
    tmp_path, labels, inc_probability, succ_probability, sum_p
):
    items = [
        {**p(i), "labels": {"p": value, "inc": int(value is None)}}
        for i, value in enumerate(labels)
    ]
    loop = Loop(tmp_path, world=World(train=[items], validation=[items[:1]])).write()
    loop.run()
    got = load_global_fit(loop.out)
    incomparable = sum(value is None for value in labels)
    assert (got.pairs, got.incomparable, got.comparable) == (
        2,
        incomparable,
        2 - incomparable,
    )
    assert (
        got.inc_probability == inc_probability
        and got.succ_probability == succ_probability
    )
    assert got.sum_p == sum_p and got.inc_logit is None
    if succ_probability is None:
        assert got.succ_logit is None
    else:
        assert got.succ_logit == 0


def shard_bytes(tmp_path, value: GlobalFit) -> bytes:
    source = tmp_path / "source"
    write_global_fit(source, value)
    return (source / "global-fit.json").read_bytes()


def put_shard_zero(out_dir, data: bytes, staged=False) -> None:
    directory = out_dir / "reference" / ("000000.part" if staged else "000000")
    directory.mkdir(parents=True)
    (directory / "baselines.json").write_bytes(data)


def test_load_global_fit_shared_source_and_staged_shard_precedence(tmp_path):
    value = fit()
    write_global_fit(tmp_path, value)
    original = (tmp_path / "global-fit.json").read_bytes()
    put_shard_zero(tmp_path, b"not json", staged=True)
    assert load_global_fit(tmp_path) == value
    assert (tmp_path / "global-fit.json").read_bytes() == original


def test_load_global_fit_complete_shard_source_and_matching_shared_source(tmp_path):
    value = fit(inc_probability=0.5)
    put_shard_zero(tmp_path, shard_bytes(tmp_path, value))
    assert load_global_fit(tmp_path) == value
    assert not (tmp_path / "global-fit.json").exists()
    write_global_fit(tmp_path, value)
    assert load_global_fit(tmp_path) == value


def test_load_global_fit_mismatch_raises(tmp_path):
    put_shard_zero(tmp_path, shard_bytes(tmp_path, fit()))
    write_global_fit(tmp_path, dataclasses.replace(fit(), pairs=99))
    with pytest.raises(RuntimeError):
        load_global_fit(tmp_path)


def test_resume_reuses_shared_fit_without_an_index_scan(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    original = (loop.out / "global-fit.json").read_bytes()
    (loop.out / "model.pt").unlink()
    for part in ("train", "validation"):
        for path in (loop.repo / "w/index" / part).rglob("items.json"):
            path.unlink()
    result = loop.run()
    assert (
        result.error is None and (loop.out / "global-fit.json").read_bytes() == original
    )


def test_selection_largest_remainder_ties_offsets_order_and_local_rng(tmp_path):
    import numpy as np

    world = labelled_world([[0.0, 1.0, None], [None, None, None]],
                           [[0.0, 0.0, 1.0], [None, None, None]])  # fmt: skip
    loop = Loop(tmp_path, world=world, reference_pairs=2).write()
    schedule_rng = np.random.Generator(np.random.PCG64([5, 3]))
    before = schedule_rng.bit_generator.state
    (selection, fitted), _, _ = select(loop)
    want_selection, want_fit = selection_of(world, 5, 2)
    assert selection == want_selection and fitted == want_fit
    assert schedule_rng.bit_generator.state == before
    assert [item["part"] for item in selection["items"]] == [
        "train",
        "train",
        "validation",
        "validation",
    ]
    assert "1.0" not in {item["stratum"] for item in selection["items"]}
    ordinals = [
        (item["part"], item["stratum"], item["shard"], item["item_index"])
        for item in selection["items"]
    ]
    assert len(ordinals) == len(set(ordinals))


def test_selection_two_scans_shared_fit_one_scan_complete_shard_none(
    tmp_path, monkeypatch, capsys
):
    from progress_bars import Bar, screen, tqdm_defaults

    tqdm_defaults(monkeypatch)
    loop = Loop(tmp_path).write()
    reads = count_item_reads(monkeypatch)
    (selection, fitted), _, _ = select(loop)
    assert len(reads) == 8
    assert screen(capsys.readouterr().err) == [Bar("pretrain reference counts", 4, 4, "shard"),
                                                Bar("pretrain reference select", 4, 4, "shard")]  # fmt: skip
    select(loop)
    assert len(reads) == 12
    assert screen(capsys.readouterr().err) == ["pretrain reference counts: already done",
                                                Bar("pretrain reference select", 4, 4, "shard")]  # fmt: skip
    write_selection(loop.out / "reference/000000", selection, fitted)
    select(loop)
    assert len(reads) == 12
    assert screen(capsys.readouterr().err) == ["pretrain reference counts: already done",
                                                "pretrain reference select: already done"]  # fmt: skip


def test_crash_before_global_fit_repeats_both_passes(tmp_path, monkeypatch):
    loop = Loop(tmp_path).write()
    count_item_reads(monkeypatch, fail_at=2)
    with pytest.raises(OSError, match="injected"):
        select(loop)
    assert not (loop.out / "global-fit.json").exists()
    monkeypatch.undo()
    reads = count_item_reads(monkeypatch)
    select(loop)
    assert len(reads) == 8


def test_crash_after_global_fit_repeats_only_selection_pass(tmp_path, monkeypatch):
    loop = Loop(tmp_path).write()
    count_item_reads(monkeypatch, fail_at=5)
    with pytest.raises(OSError, match="injected"):
        select(loop)
    assert (loop.out / "global-fit.json").exists()
    monkeypatch.undo()
    reads = count_item_reads(monkeypatch)
    select(loop)
    assert len(reads) == 4


def test_insufficient_validation_aborts_before_second_pass(tmp_path, monkeypatch):
    loop = Loop(tmp_path, reference_pairs=2).write()
    reads = count_item_reads(monkeypatch)
    with pytest.raises(
        RuntimeError, match="^validation has only 1 pairs, reference needs 2$"
    ):
        select(loop)
    assert len(reads) == 4


def bce(x: float, y: float) -> float:
    return max(x, 0) - x * y + math.log1p(math.exp(-abs(x)))


def expected_baselines(labels: list[dict], fitted: GlobalFit) -> dict:
    n = len(labels)
    comparable = [x["p"] for x in labels if not x["inc"]]
    u, v = fitted.inc_probability, fitted.succ_probability
    inc = (
        sum(-x["inc"] * math.log(u) - (1 - x["inc"]) * math.log(1 - u) for x in labels)
        * 2
        / n
    )
    succ = sum(-p * math.log(v) - (1 - p) * math.log(1 - v) for p in comparable) * 2 / n
    entropy = sum(-p * math.log(p) - (1 - p) * math.log(1 - p) if p not in (0, 1) else 0.0 for p in comparable) * 2 / n  # fmt: skip
    return {"zero": 2 * math.log(2) * (1 + len(comparable) / n),
            "inc_only": inc + 2 * math.log(2) * len(comparable) / n, "both": inc + succ,
            "target_entropy": entropy, "inc_constant": inc, "succ_constant": succ}  # fmt: skip


def expected_condition(module, selection: list[dict], Z0=None, phi=None) -> dict:
    from .loop_world import clip

    result = {
        part: {"s": [], "t": [], "labels": []} for part in ("train", "validation")
    }
    for j, item in enumerate(selection):
        if phi is None:
            za, zb = Z0[j, 0:1], Z0[j, 1:2]
        else:
            z = phi([clip(item["a"]), clip(item["b"])])
            za, zb = z[0:1], z[1:2]
        s, t = module(za, zb)
        part = result[item["part"]]
        part["s"].append(float(s[0].detach()))
        part["t"].append(float(t[0].detach()))
        part["labels"].append(item["labels"])
    return result


def assert_condition(got: dict, expected: dict, fitted: GlobalFit) -> None:
    for part, values in expected.items():
        labels = values["labels"]
        n = len(labels)
        losses = {"P_inc": sum(2 * bce(t, y["inc"]) for t, y in zip(values["t"], labels)) / n,
                  "P_succ": sum(2 * bce(s, y["p"]) for s, y in zip(values["s"], labels) if not y["inc"]) / n}  # fmt: skip
        losses["P"] = losses["P_inc"] + losses["P_succ"]
        for key, value in losses.items():
            assert got[part]["losses"][key] == pytest.approx(value, rel=1e-9, abs=1e-10)
        assert got[part]["preference"]["inc"]["mean"] == pytest.approx(sum(1 / (1 + math.exp(-x)) for x in values["t"]) / n, rel=1e-7)  # fmt: skip
        comparable = [x for x, y in zip(values["s"], labels) if not y["inc"]]
        assert got[part]["preference"]["succ"]["mean"] == pytest.approx(sum(1 / (1 + math.exp(-x)) for x in comparable) / len(comparable), rel=1e-7)  # fmt: skip
        for key, value in expected_baselines(labels, fitted).items():
            assert got[part]["baselines"][key] == pytest.approx(
                value, rel=1e-9, abs=1e-10
            )


def test_initial_artifacts_and_four_conditions_after_update(
    tmp_path, monkeypatch, capsys
):
    import copy

    import torch
    from progress_bars import Bar, screen, tqdm_defaults

    from mcpref.pretrain.reference import load_initial, write_initial
    from mcpref.pretrain.reference_eval import evaluate_initial, evaluate_reference

    from .index_run import repo_root
    from .loop_modules import Pref, fakes
    from .loop_world import clip

    tqdm_defaults(monkeypatch)
    loop = Loop(tmp_path, world=full_world(), reference_pairs=64, batch=64, micro_batch=64, epochs=1).write()  # fmt: skip
    (selection, fitted), cfg, run = select(loop)
    capsys.readouterr()
    mods = fakes(noise=False, stateful=False)
    with repo_root(loop.repo):
        initial, metrics = evaluate_initial(cfg, mods, selection, fitted, run)
    assert screen(capsys.readouterr().err) == [
        Bar("pretrain reference initial", 128, 128, "pair")
    ]
    assert mods["phi"].calls == mods["P"].calls == 128
    write_initial(loop.out / "reference", selection, fitted, initial, metrics)
    reference = load_initial(loop.out / "reference")
    assert reference.selection == selection and reference.baselines == fitted
    assert reference.initial_metrics == metrics
    assert reference.Z0.shape == (128, 2, 3, 2) and reference.Z0.device.type == "cpu"
    assert reference.Z0.dtype == torch.float64 and not reference.Z0.requires_grad
    oracle = fakes(noise=False, stateful=False)
    expected_z = torch.stack(
        [
            oracle["phi"]([clip(item["a"]), clip(item["b"])])
            for item in selection["items"]
        ]
    )
    torch.testing.assert_close(reference.Z0, expected_z)
    assert set(reference.P0) == {"u", "beta", "v"}
    for key, value in oracle["P"].state_dict().items():
        assert reference.P0[key].device.type == "cpu" and reference.P0[key].equal(value)
    assert (loop.out / "reference/000000/baselines.json").read_bytes() == (
        loop.out / "global-fit.json"
    ).read_bytes()
    assert not (loop.out / "reference/000000.part").exists()
    for condition in ("P_only", "phi_only", "initial"):
        for part in ("train", "validation"):
            assert (
                metrics["conditions"][condition][part]
                == metrics["conditions"]["initial"][part]
            )
            assert (
                "pi" not in metrics["conditions"][condition][part]
                and "phi" not in metrics["conditions"][condition][part]
            )
    for part in ("train", "validation"):
        joint = metrics["conditions"]["joint"][part]
        initial_part = metrics["conditions"]["initial"][part]
        assert (
            joint["preference"] == initial_part["preference"]
            and joint["baselines"] == initial_part["baselines"]
        )
        assert {
            key: joint["losses"][key] for key in ("P", "P_inc", "P_succ")
        } == initial_part["losses"]
    assert set(metrics["conditions"]) == {"joint", "P_only", "phi_only", "initial"}
    assert set(metrics["conditions"]["joint"]["train"]["losses"]) == {"P", "P_inc", "P_succ", "phi", "phi_push", "phi_constraint", "phi_dual", "phi_norm", "pi", "pi_action", "pi_shift"}  # fmt: skip

    with torch.no_grad():
        mods["phi"].A.add_(0.03125)
        mods["P"].u.add_(0.0625)
        mods["P"].beta.sub_(0.125)
        mods["P"].v.add_(0.25)
    from .test_reference_resume import nested_equal, optimizer_of, snapshot

    optimizer = optimizer_of(mods)
    state_before = snapshot(mods, optimizer)
    before = {key: value.clone() for key, value in mods["P"].state_dict().items()}
    phi_calls, p_calls = mods["phi"].calls, mods["P"].calls
    with repo_root(loop.repo):
        got = evaluate_reference(cfg, mods, reference, run, 500)
    assert nested_equal(snapshot(mods, optimizer), state_before)
    assert mods["phi"].calls - phi_calls == 128 and mods["P"].calls - p_calls == 256
    assert all(
        before[key].equal(value) for key, value in mods["P"].state_dict().items()
    )
    oracle_mods = copy.deepcopy(mods)
    frozen = Pref()
    frozen.load_state_dict(reference.P0)
    expected = {
        "joint": expected_condition(
            oracle_mods["P"], selection["items"], phi=oracle_mods["phi"]
        ),
        "P_only": expected_condition(
            oracle_mods["P"], selection["items"], Z0=reference.Z0
        ),
        "phi_only": expected_condition(
            frozen, selection["items"], phi=oracle_mods["phi"]
        ),
    }
    for condition, values in expected.items():
        assert_condition(got["conditions"][condition], values, fitted)
    assert got["conditions"]["initial"] == metrics["conditions"]["initial"]
    assert got["step"] == 500 and got["version"] == 1
    for condition in ("P_only", "phi_only", "initial"):
        assert set(got["conditions"][condition]["train"]["losses"]) == {
            "P",
            "P_inc",
            "P_succ",
        }
