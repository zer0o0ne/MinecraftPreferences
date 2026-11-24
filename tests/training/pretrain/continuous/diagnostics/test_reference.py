"""Continuous reference prefixes, frozen conditions, bounded state and RNG isolation."""

import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from mcpref.training.pretrain.continuous.diagnostics import reference_replay
from mcpref.training.pretrain.continuous.diagnostics.reference import (
    evaluate_initial,
    evaluate_reference,
    select_reference,
)


@pytest.mark.parametrize("raised", [False, True])
def test_autocast_cache_does_not_cross_reference_or_training_updates(raised):
    cfg = SimpleNamespace(device="cpu")
    module = torch.nn.Linear(3, 2).train()
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    rng = torch.get_rng_state()
    for value in (1.0, 2.0):
        with torch.no_grad():
            module.weight.fill_(value)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = module(inputs)
        try:
            with reference_replay.evaluation_state(cfg, {"phi": module}):
                assert not module.training
                assert not torch.is_autocast_enabled("cpu")
                for _ in range(2):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        actual = module(inputs)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.rand(2)
                if raised:
                    raise RuntimeError("reference failed")
        except RuntimeError as exc:
            assert raised and str(exc) == "reference failed"
        assert module.training
        assert torch.equal(torch.get_rng_state(), rng)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            module(inputs).sum().backward()
        torch.testing.assert_close(module.weight.grad, inputs.expand(2, -1))
        module.zero_grad(set_to_none=True)


class Phi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout = torch.nn.Dropout(0.5)
        self.register_buffer("visits", torch.tensor(0))
        self.cfg, self.sprite = SimpleNamespace(), None


class Preference(torch.nn.Module):
    history = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, A, B):
        return self.weight * (A.mean((1, 2)) - B.mean((1, 2))), self.weight * (
            A.mean((1, 2)) + B.mean((1, 2))
        )


class Loss(torch.nn.Module):
    def terms(self, output, labels):
        s, t = output
        return {"inc": t.square(), "succ": s.square(), "total": s.square() + t.square()}


def world(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE edges(id TEXT,a TEXT,b TEXT,part TEXT,body TEXT); CREATE TABLE endpoints(id TEXT,fragment TEXT,completion TEXT); CREATE TABLE fragments(id TEXT,part TEXT,body TEXT);"
    )
    data, visits = {}, []
    for part in ("train", "validation"):
        for j in range(2):
            key = f"{part}{j}"
            fragment = {"id": key, "observations": 4}
            db.execute(
                "INSERT INTO fragments VALUES(?,?,?)", (key, part, json.dumps(fragment))
            )
            db.execute("INSERT INTO endpoints VALUES(?,?,?)", (key, key, f"{key}:2"))
            data[key] = [
                SimpleNamespace(
                    id=f"{key}:{i}",
                    frame=float(i + 1 + j),
                    line={},
                    meta="m",
                    interval=None,
                    endpoints=(
                        {"id": key, "observation_ids": [f"{key}:1", f"{key}:2"]},
                    )
                    if i == 2
                    else (),
                )
                for i in range(4)
            ]
        edge = {
            "id": part,
            "a": f"{part}0",
            "b": f"{part}1",
            "labels": {"inc": 0, "p": 0.5},
        }
        db.execute(
            "INSERT INTO edges VALUES(?,?,?,?,?)",
            (part, edge["a"], edge["b"], part, json.dumps(edge)),
        )

    class Reader:
        def __init__(self, *args):
            pass

        def observations(self, fragment):
            for row in data[fragment["id"]]:
                visits.append(row.id)
                yield row

    def step(phi, frame, line, state):
        phi.visits.add_(1)
        torch.rand(1)
        z = (phi.weight * frame + (0 if state is None else state)).reshape(1)
        return z, z

    monkeypatch.setattr(reference_replay, "StreamReader", Reader)
    monkeypatch.setattr(reference_replay, "observation_step", step)
    monkeypatch.setattr(reference_replay, "own_losses", lambda *args: {})
    phi = Phi().train()
    phi.dropout.eval()
    phi_loss = torch.nn.Module()
    phi_loss.register_parameter("log_lambda", torch.nn.Parameter(torch.tensor(0.0)))
    modules = {"phi": phi, "phi_loss": phi_loss, "P": Preference(), "P_loss": Loss()}
    cfg = SimpleNamespace(
        seed=7,
        reference_pairs=1,
        device="cpu",
        calibration_bins=4,
        action_copy_eps=0.1,
        saturation_threshold=0.01,
        inc_classification_threshold=0.5,
        quantiles=(0.5,),
    )
    return cfg, modules, SimpleNamespace(db=db), visits


def test_prefixes_begin_at_fragment_start_stop_after_completion_and_preserve_state(
    monkeypatch,
):
    cfg, modules, index, visits = world(monkeypatch)
    selected, fit = select_reference(cfg, index)
    assert select_reference(cfg, index) == (selected, fit)
    rng = torch.get_rng_state()
    initial, metrics, pairs, _ = evaluate_initial(cfg, modules, index, selected, fit)
    assert visits == [
        f"{part}{j}:{i}"
        for part in ("train", "validation")
        for j in range(2)
        for i in range(3)
    ]
    torch.testing.assert_close(initial["Z0"]["train0"], torch.tensor([[3.0], [6.0]]))
    torch.testing.assert_close(initial["Z0"]["train1"], torch.tensor([[5.0], [9.0]]))
    assert modules["phi"].training and not modules["phi"].dropout.training
    assert modules["phi"].visits.item() == 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert metrics["conditions"]["joint"]["train"]["counts"]["P"] == 1
    assert metrics["conditions"]["joint"]["train"]["losses"]["P"] == 138.5
    assert all(pair.baseline == pair.joint_post == pair.only_phi_post for pair in pairs)


def test_current_initial_phi_and_heads_are_separated_and_replay_is_fresh(monkeypatch):
    cfg, modules, index, visits = world(monkeypatch)
    selected, fit = select_reference(cfg, index)
    initial, _, _, _ = evaluate_initial(cfg, modules, index, selected, fit)
    with torch.no_grad():
        modules["phi"].weight.fill_(2)
        modules["P"].weight.fill_(3)
    visits.clear()
    current, pairs, _ = evaluate_reference(
        cfg, modules, index, selected, fit, initial, 5
    )
    losses = {
        condition: parts["train"]["losses"]["P"]
        for condition, parts in current["conditions"].items()
    }
    assert losses == {
        "joint": 36 * 138.5,
        "P_only": 9 * 138.5,
        "phi_only": 4 * 138.5,
        "initial": 138.5,
    }
    assert len(visits) == 12
    assert pairs[0].delta_only_phi == 3 * 138.5
    assert pairs[0].delta_joint == 35 * 138.5
    assert all(not value.requires_grad for value in initial["Z0"].values())
