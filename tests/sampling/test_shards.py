"""Primitive step/state round trips and cross-shard raw history of env-sampling.md."""

import dataclasses

import numpy as np
import torch
from mcts.helpers import latent_history, node
from transformers import DynamicCache

from mcpref.mcts.tree import Tree
from mcpref.sampling.shards import (
    RawHistory,
    SamplerState,
    StepRecord,
    load_state,
    load_step,
    read_history,
    write_state,
    write_step,
)

from .helpers import FakePhi, FakeStream, continuation, state


def tree_mapping():
    return {
        "root": {
            "z": torch.tensor([0.0, 0.5]),
            "x": torch.tensor([0.0, 0.5]),
            "action": None,
            "prior": None,
            "shift": None,
            "children": [
                {
                    "z": torch.tensor([0.1, 0.7]),
                    "x": torch.tensor([0.11, 0.72]),
                    "action": 1,
                    "prior": 0.25,
                    "shift": torch.tensor([0.1, 0.2]),
                    "children": [],
                    "n": 2,
                    "mean": 0.3,
                    "value": -0.4,
                }
            ],
            "n": 0,
            "mean": 0.0,
            "value": 0.0,
        },
        "base": {
            "Y": torch.zeros(1, 1, 2, dtype=torch.float64),
            "C": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
        },
    }


def record(index: int) -> StepRecord:
    return StepRecord(
        index=index,
        before=state(index),
        after=state(index + 1),
        action=index % 3,
        probabilities=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        delta=torch.tensor([1.0, 1.0], dtype=torch.float64),
        predicted=torch.tensor([0.1, 0.2], dtype=torch.float64),
        elapsed_ms=12.5,
        preference_close={"succ": 1.0, "inc": 2.0},
        preference_start={"succ": 3.0, "inc": 4.0},
        tree=tree_mapping(),
    )


def test_record_interfaces_and_primitive_step_round_trip(tmp_path, monkeypatch):
    assert [field.name for field in dataclasses.fields(StepRecord)] == [
        "index", "before", "after", "action", "probabilities", "delta", "predicted",
        "elapsed_ms", "preference_close", "preference_start", "tree",
    ]  # fmt: skip
    out = tmp_path / "out"
    shard = out / "shards/000000"
    shard.mkdir(parents=True)
    before = dataclasses.replace(
        state(0), gui=True, cls="inventory", hotbar=2,
        mask=np.array([True, False, True]), line={"id": 0, "nested": {"x": 1.5}},
    )  # fmt: skip
    original = dataclasses.replace(record(0), before=before)
    write_step(shard, 0, original)
    write_state(shard, sampler_state(FakePhi()))
    path = shard / "steps/000000.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {field.name for field in dataclasses.fields(StepRecord)}
    assert isinstance(raw["before"]["frame"], torch.Tensor)
    assert raw["before"]["frame"].dtype == torch.uint8
    assert raw["probabilities"].dtype == torch.float64
    loads = []
    real_load = torch.load

    def load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("mcpref.sampling.shards.torch.load", load)
    got = load_step(out, 0)
    assert loads == [{"map_location": "cpu", "weights_only": True}] * 2
    assert got.index == 0 and isinstance(got.before.frame, np.ndarray)
    np.testing.assert_array_equal(got.before.frame, state(0).frame)
    assert (got.before.gui, got.before.cls, got.before.hotbar) == (
        True,
        "inventory",
        2,
    )
    assert got.before.line == {"id": 0, "nested": {"x": 1.5}}
    np.testing.assert_array_equal(got.before.mask, [True, False, True])
    torch.testing.assert_close(got.probabilities, record(0).probabilities)
    assert got.preference_close == {"succ": 1.0, "inc": 2.0}
    assert got.tree.keys() == raw["tree"].keys()
    torch.testing.assert_close(got.tree["root"]["z"], raw["tree"]["root"]["z"])
    torch.testing.assert_close(
        got.tree["root"]["children"][0]["shift"],
        raw["tree"]["root"]["children"][0]["shift"],
    )


def sampler_state(phi: FakePhi) -> SamplerState:
    real = latent_history([[2.0, 2.5]])
    tree = Tree(real)
    child = node([2.1, 2.7], action=1, prior=0.5, parent=tree.root, n=2, mean=0.3)
    grandchild = node([2.2, 2.9], action=2, prior=0.25, parent=child, value=-0.2)
    child.children = [grandchild]
    tree.root.children = [child]
    stream = FakeStream(phi, 2)
    stream.ids, stream.grids, stream.first = [2], [torch.ones(1, 3)], {"id": 2}
    stream.cache = DynamicCache(
        ddp_cache_data=[(torch.ones(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), None)],
        config=phi.backbone.config,
    )
    return SamplerState(
        observation=state(2),
        env=continuation(held_action=1),
        history=real,
        tree=tree,
        rng_state=np.random.default_rng(7).bit_generator.state,
        next_step=2,
        stream=stream,
        fingerprint="fake-model",
    )


def test_state_round_trip_restores_tree_links_stream_and_fields(tmp_path):
    assert [field.name for field in dataclasses.fields(SamplerState)] == [
        "observation", "env", "history", "tree", "rng_state", "next_step", "stream",
        "fingerprint",
    ]  # fmt: skip
    assert [field.name for field in dataclasses.fields(RawHistory)] == [
        "states",
        "actions",
        "restart",
    ]
    phi = FakePhi()
    out = tmp_path / "out"
    shard = out / "shards/000000"
    shard.mkdir(parents=True)
    write_state(shard, sampler_state(phi))
    raw = torch.load(shard / "state.pt", map_location="cpu", weights_only=True)
    assert set(raw) == {
        "observation", "env", "history", "tree", "rng", "next_step", "stream",
        "fingerprint",
    }  # fmt: skip
    assert "parent" not in raw["tree"]["root"]
    assert raw["env"] == {
        "info": {
            "isGuiOpen": True,
            "custom": {"jump": 2},
            "mine_block": {"stone": 3},
            "use_item": {"torch": 4},
        },
        "cursor": (12.5, 34.25),
        "held_action": 1,
    }
    got = load_state(out, phi, 2, torch.device("cpu"))
    assert got.next_step == 2 and got.fingerprint == "fake-model"
    assert got.rng_state == sampler_state(phi).rng_state
    np.testing.assert_array_equal(got.observation.frame, state(2).frame)
    assert got.env == continuation(held_action=1)
    assert isinstance(got.env.cursor, tuple)
    torch.testing.assert_close(got.history.Y, sampler_state(phi).history.Y)
    child = got.tree.root.children[0]
    assert child.parent is got.tree.root and child.children[0].parent is child
    assert (child.n, child.mean, child.children[0].value) == (2, 0.3, -0.2)
    assert got.stream.ids == [2] and got.stream.first == {"id": 2}


def test_raw_history_boundaries_and_cross_shard_reader(tmp_path):
    out = tmp_path / "out"
    for k in range(3):
        shard = out / f"shards/{k:06d}"
        shard.mkdir(parents=True)
        for j in range(2):
            write_step(shard, j, record(2 * k + j))
    write_state(out / "shards/000000", sampler_state(FakePhi()))
    cases = {
        0: (0, [0], [None]),
        2: (0, [0, 1, 2], [None, 0, 1]),
        3: (0, [0, 1, 2, 3], [None, 0, 1, 2]),
        4: (0, [0, 1, 2, 3, 4], [None, 0, 1, 2, 0]),
        5: (3, [3, 4, 5], [2, 0, 1]),
    }
    for index, (restart, states, actions) in cases.items():
        got = read_history(out, index, 3)
        assert got.restart == restart
        assert [item.line["id"] for item in got.states] == states
        assert got.actions == actions
