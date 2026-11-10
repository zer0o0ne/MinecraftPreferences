"""Reference evaluation state, cadence and resume oracles of specs/pretrain-diagnostics.md."""

import contextlib
import copy
import dataclasses
import json
import shutil

import pytest
import torch

from mcpref.training.pretrain.references.eval import evaluation_state
from training.pretrain.loop_run import Loop
from training.pretrain.steps.test_loss import CFG


def nested_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.equal(b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def snapshot(modules: dict, optimizer: torch.optim.Optimizer) -> dict:
    return {
        "modules": {
            name: {key: value.clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()
        },
        "gradients": {
            name: {
                key: None if p.grad is None else p.grad.clone()
                for key, p in module.named_parameters()
            }
            for name, module in modules.items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state(),
    }


def optimizer_of(modules: dict) -> torch.optim.AdamW:
    groups = [
        group for module in modules.values() for group in module.groups().values()
    ]
    optimizer = torch.optim.AdamW([{"params": group, "lr": 0.01} for group in groups])
    for group in groups:
        for parameter in group:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return optimizer


@pytest.mark.parametrize("raised", [False, True], ids=["return", "exception"])
def test_evaluation_state_preserves_everything_on_return_and_exception(raised):
    from mcpref.training.pretrain.references.eval import evaluation_state
    from training.pretrain.loop_modules import fakes

    modules = fakes(noise=False)
    for i, module in enumerate(modules.values()):
        module.train(i % 2 == 0)
    modules["phi_loss"].seen.fill_(7)
    optimizer = optimizer_of(modules)
    before = snapshot(modules, optimizer)
    modes = {name: module.training for name, module in modules.items()}

    class Marker(Exception):
        pass

    context = pytest.raises(Marker) if raised else contextlib.nullcontext()
    with context, evaluation_state(CFG, modules):
        assert all(not module.training for module in modules.values())
        assert not torch.is_grad_enabled()
        torch.rand(11)
        if raised:
            raise Marker
    after = snapshot(modules, optimizer)
    assert {name: module.training for name, module in modules.items()} == modes
    assert nested_equal(after, before)


def test_policy_dropout_disabled_and_training_mode_restored(tmp_path, masks):
    from models.policy.policy_oracle import cfg_of

    from mcpref.models.policy.network import Policy
    from training.pretrain.loop_world import clip, u

    policy = Policy(cfg_of(history=3, width=4, heads=2, hidden=6, shift_hidden=5, action_dropout=0.5), 2, masks.names).double()  # fmt: skip
    policy.train()
    clips = [clip(u(0)["clip"])]
    z = torch.randn(1, 3, 2, dtype=torch.float64)
    with evaluation_state(CFG, {"pi": policy}):
        rng = torch.get_rng_state()
        first = policy(z, clips)
        second = policy(z, clips)
        assert torch.equal(rng, torch.get_rng_state())
        assert first[0].equal(second[0]) and first[1].equal(second[1])
    assert policy.training
    rng = torch.get_rng_state()
    policy(z, clips)
    assert not torch.get_rng_state().equal(rng)


def reference_function():
    from mcpref.training.pretrain.steps import steps

    return steps, steps.evaluate_reference_event


def cadence_loop(tmp_path, keep=1):
    return Loop(tmp_path, batch=3, micro_batch=2, reference_steps=5, checkpoint_steps=1,
                reference_pairs=1, keep_checkpoints=keep).write()  # fmt: skip


def test_cadence_checkpoint_first_call_counts_and_no_off_cadence_final(tmp_path):
    loop = cadence_loop(tmp_path)
    mods = loop.modules()
    target, real = reference_function()
    calls = []

    def wrapped(cfg, modules, reference, run, step):
        calls.append((step, (loop.checkpoint_dir(4) / "losses.json").exists()))
        return real(cfg, modules, reference, run, step)

    result = loop.run(
        mods, patch=lambda mp: mp.setattr(target, "evaluate_reference_event", wrapped)
    )
    from progress_bars import Bar, screen

    assert Bar("pretrain reference", 2, 2, "pair") in screen(result.err)
    assert calls == [(5, True)]
    assert sorted(path.name for path in (loop.out / "reference").iterdir()) == [
        "000000",
        "000001",
    ]
    metrics = json.loads((loop.out / "reference/000001/metrics.json").read_text())
    later_pairs = json.loads((loop.out / "reference/000001/pairs.json").read_text())
    assert metrics["step"] == 5 and not (loop.out / "reference/000002").exists()
    assert len(later_pairs) == 2
    assert all(
        pair["delta_joint"] == pair["joint_post"] - pair["baseline"]
        for pair in later_pairs
    )
    assert all(
        pair["delta_only_phi"] == pair["only_phi_post"] - pair["baseline"]
        for pair in later_pairs
    )
    diagnostic_records = [
        json.loads(path.read_text())
        for path in (loop.repo / "w/diagnostics/pretrain").glob("*/metrics.json")
    ]
    late_initial = next(
        record
        for record in diagnostic_records
        if record["id"] == "reference:1:train:initial"
    )
    assert (
        late_initial["joined"]["condition"] == metrics["conditions"]["initial"]["train"]
    )
    assert late_initial["joined"]["condition"]["losses"]["P"] is not None
    assert late_initial["metrics"]["P"]["n"] == 0
    assert late_initial["metrics"]["P"]["reason"] == "empty_pair_population"
    assert (
        mods["phi"].calls == 20
    )  # 2 initial + 10 training + 2 due pairs + 6 selected counterfactual calls


def test_positive_reference_progress_has_creation_and_close_states(tmp_path):
    from progress_bars import Bar, assert_bars

    result = cadence_loop(tmp_path).run()
    assert_bars(result.err, [Bar("pretrain reference", 2, 2, "pair")])


def test_resume_recomputes_only_missing_latest_reference(tmp_path):
    from training.pretrain.data.index_run import Interrupted

    whole = cadence_loop(tmp_path / "whole")
    whole.run()
    cut = cadence_loop(tmp_path / "cut")
    target, _ = reference_function()

    def interrupt(cfg, modules, reference, run, step):
        raise Interrupted("before reference publication")

    cut.run(
        patch=lambda mp: mp.setattr(target, "evaluate_reference_event", interrupt),
        raises=Interrupted,
    )
    assert (cut.checkpoint_dir(4) / "losses.json").exists()
    assert not (cut.out / "model.pt").exists()
    assert sorted(path.name for path in (cut.out / "reference").iterdir()) == ["000000"]
    mods = cut.modules()
    cut.run(mods)
    import math

    import numpy as np

    from training.schedule_oracle import epoch

    rng = np.random.Generator(np.random.PCG64([cut.config["seed"], 3]))
    sizes = [len(shard) for shard in cut.world.train]
    entries = [
        entry
        for _ in range(cut.config["epochs"])
        for entry in epoch(rng, sizes, cut.config["batch"])
    ]
    remaining = sum(
        math.ceil(len(pos) / cut.config["micro_batch"]) for _, pos in entries[5:]
    )
    selection = json.loads((cut.out / "gradient-flow/selection.json").read_text())
    selected = sum(entry["step"] > 5 for entry in selection["steps"])
    assert mods["phi"].calls == 2 + remaining + 2 * selected
    for shard in ("000000", "000001"):
        for name in (
            "selection.json",
            "baselines.json",
            "initial.pt",
            "metrics.json",
            "pairs.json",
        ):
            if shard == "000001" and name not in ("metrics.json", "pairs.json"):
                continue
            assert (cut.out / "reference" / shard / name).read_bytes() == (
                whole.out / "reference" / shard / name
            ).read_bytes()
    assert not (cut.out / "reference/000002").exists()


def test_resume_after_publication_reuses_shard_zero_verbatim(tmp_path):
    loop = cadence_loop(tmp_path)
    loop.run()
    before = loop.tree()
    (loop.out / "model.pt").unlink()
    mods = loop.modules()
    loop.run(mods)
    assert all(module.calls == 0 for module in mods.values())
    after = loop.tree()
    for name in before:
        if name != "model.pt":
            assert after[name] == before[name]


def test_resume_at_completed_due_step_prints_already_done_without_reevaluation(
    tmp_path,
):
    from progress_bars import screen

    loop = cadence_loop(tmp_path, keep=20)
    loop.run()
    (loop.out / "model.pt").unlink()
    shutil.rmtree(loop.checkpoint_dir(5))
    shutil.rmtree(loop.repo / "w/diagnostics")
    before = (loop.out / "reference/000001/metrics.json").read_bytes()
    mods = loop.modules()
    result = loop.run(mods)
    assert "pretrain reference: already done" in screen(result.err)
    assert (loop.out / "reference/000001/metrics.json").read_bytes() == before
    assert mods["phi"].calls > 0 and not (loop.out / "reference/000002").exists()


@pytest.mark.parametrize("problem", ["gap", "future"])
def test_resume_rejects_reference_gap_or_future_shard(tmp_path, problem):
    loop = cadence_loop(tmp_path)
    loop.run()
    (loop.out / "model.pt").unlink()
    if problem == "gap":
        shutil.rmtree(loop.out / "reference/000001")
    else:
        shutil.copytree(loop.out / "reference/000001", loop.out / "reference/000002")
    with pytest.raises(RuntimeError):
        loop.run()


def test_legacy_shard_zero_resumes_metrics_only_without_backfill(tmp_path):
    loop = cadence_loop(tmp_path, keep=20)
    loop.run()
    (loop.out / "reference/000000/pairs.json").unlink()
    shutil.rmtree(loop.out / "reference/000001")
    shutil.rmtree(loop.repo / "w/diagnostics")
    shutil.rmtree(loop.checkpoint_dir(5))
    (loop.out / "model.pt").unlink()
    result = loop.run(loop.modules())
    assert result.error is None
    assert (loop.out / "reference/000001/metrics.json").is_file()
    assert not list((loop.out / "reference").rglob("pairs.json"))

    from mcpref.training.pretrain.references.reference import load_initial
    from training.pretrain.references.reference_oracle import select

    reference = load_initial(loop.out / "reference")
    assert reference.initial_pairs is None
    (_, _), cfg, run = select(loop)
    from mcpref.training.pretrain.references.eval import evaluate_reference
    from training.pretrain.data.index_run import repo_root

    with repo_root(loop.repo):
        metrics, pairs = evaluate_reference(cfg, loop.modules(), reference, run, 5)
    assert pairs is None
    initial = json.loads((loop.out / "reference/000000/metrics.json").read_text())
    assert metrics["conditions"]["initial"] == initial["conditions"]["initial"]


@pytest.mark.parametrize("problem", ["new_missing", "legacy_extra"])
def test_resume_rejects_mixed_reference_pair_layouts(tmp_path, problem):
    loop = cadence_loop(tmp_path)
    loop.run()
    (loop.out / "model.pt").unlink()
    shard = "000001" if problem == "new_missing" else "000000"
    (loop.out / "reference" / shard / "pairs.json").unlink()
    with pytest.raises(RuntimeError):
        loop.run()


@pytest.mark.gpu
def test_evaluation_state_restores_configured_cuda_rng(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    index = 1 if torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{index}")
    other = torch.device("cuda:0")
    module = torch.nn.Dropout().to(device)
    cfg = dataclasses.replace(CFG, device=str(device))
    cpu = torch.get_rng_state()
    configured = torch.cuda.get_rng_state(device)
    other_before = torch.cuda.get_rng_state(other)
    real_get, real_set = torch.cuda.get_rng_state, torch.cuda.set_rng_state
    gets, sets = [], []

    def get(device=None):
        gets.append(device)
        return real_get(device)

    def set(state, device=None):
        sets.append(device)
        return real_set(state, device)

    monkeypatch.setattr(torch.cuda, "get_rng_state", get)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: (_ for _ in ()).throw(AssertionError("set_device called")),
    )
    gets.clear()
    with evaluation_state(cfg, {"pi": module}):
        torch.rand(3)
        torch.rand(3, device=device)
    assert torch.get_rng_state().equal(cpu)
    assert real_get(device).equal(configured)
    assert real_get(other).equal(other_before)
    assert all(torch.device(value) == device for value in gets if value is not None)
    assert sets and all(
        torch.device(value) == device for value in sets if value is not None
    )
