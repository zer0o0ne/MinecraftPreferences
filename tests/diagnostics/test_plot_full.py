"""Every block-33 plot group from accepted producer records, including curves."""

import json
from types import SimpleNamespace

import numpy as np
import torch

from diagnostics.core.world import config, context
from diagnostics.test_plot import fake_render, make_records, write_records
from mcpref.diagnostics.core.artifacts import write_record
from mcpref.diagnostics.core.observations import (
    EpisodeArtifacts,
    MemoryEvent,
    PolicyObservations,
    PpoObservations,
    ShiftObservations,
    UpdateObservations,
)
from mcpref.diagnostics.core.types import ProducerContext
from mcpref.diagnostics.plot import main
from mcpref.diagnostics.producers.mcts import on_step_record
from mcpref.diagnostics.producers.memory import on_memory_event
from mcpref.diagnostics.producers.milestones import on_episode_end
from mcpref.diagnostics.producers.policy import (
    on_policy_observation,
    on_ppo_observation,
    on_shift_observation,
)
from mcpref.diagnostics.producers.update import on_update_observation


def test_full_plot_registry_and_special_source_curves(tmp_path, monkeypatch):
    from mcpref.diagnostics import plot

    rendered = {}

    def capture(directory, entries, points):
        rendered.update(points)
        fake_render(directory, entries, points)

    monkeypatch.setattr(plot, "render", capture)
    ctx = context(
        "rl/000000/checkpoints/000000/losses.json",
        "policy",
        "joint",
        steps=(1,),
        clusters=("e0",),
    )
    policy = on_policy_observation(
        PolicyObservations(
            torch.tensor([[0.8, 0.2]]).log(),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, True]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            torch.tensor([0.8, 0.2]),
            1,
            0.1,
            2,
        ),
        cfg=config(),
        context=ctx,
    )
    ppo = on_ppo_observation(
        PpoObservations(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([[0.8, 0.2]]),
            torch.tensor([[0.7, 0.3]]),
            torch.tensor([0]),
            0.2,
            ("fresh",),
            torch.tensor([True]),
        ),
        cfg=config(),
        context=ctx,
    )
    shift = on_shift_observation(
        ShiftObservations(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.5, 0.0]]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.5]),
            torch.tensor([0.1]),
            torch.tensor([2.0]),
            0.01,
            torch.tensor([False]),
        ),
        cfg=config(),
        context=ctx,
    )
    update_ctx = ProducerContext(
        **{**ctx.__dict__, "producer": "update", "cluster_ids": ("a0", "a1")}
    )
    update = on_update_observation(
        UpdateObservations(
            {"pi.head.weight": torch.ones(2, 2)},
            {"pi.head.weight": torch.ones(2, 2)},
            {"pi": ("pi.head.weight",)},
            0.5,
            0.01,
            1,
            torch.tensor([[0.8, 0.2]]),
            torch.tensor([True, True]),
        ),
        cfg=config(),
        context=update_ctx,
    )
    memory = on_memory_event(
        MemoryEvent(
            "publication",
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            10,
            0,
            0,
            0,
            ("e0",),
            ("e0",),
        ),
        cfg=config(),
        context=ctx,
    )
    root = {
        "z": torch.zeros(2),
        "children": [
            {"action": 0, "prior": 0.8, "x": torch.tensor([0.04, 0.0])},
            {"action": 1, "prior": 0.2, "x": torch.tensor([2.0, 0.0])},
        ],
    }
    step = SimpleNamespace(
        index=1,
        tree={"root": root},
        probabilities=torch.tensor([0.75, 0.25]),
        policy_probabilities=torch.tensor([0.8, 0.2]),
        action=0,
        delta=torch.tensor([0.04, 0.0]),
        predicted=torch.tensor([0.04, 0.0]),
        before=SimpleNamespace(mask=np.array([True, True])),
    )
    mcts_ctx = ProducerContext(
        **{**ctx.__dict__, "filters": {"alpha": [1.0, 1.0], "epsilon": 0.1, "width": 1}}
    )
    mcts = on_step_record(step, cfg=config(), context=mcts_ctx)
    milestone = on_episode_end(
        EpisodeArtifacts("e0", [{"stone_pickaxe": 1}], []), cfg=config(), context=ctx
    )
    write_records(tmp_path / "diag", zip(("targets", "pretrain"), make_records()))
    records = [policy, ppo, shift, update, memory]
    for index, record in enumerate(records):
        write_record(tmp_path / "diag", "rl", index, record)
    write_record(tmp_path / "diag", "mcts", 0, mcts)
    write_record(tmp_path / "diag", "milestones", 0, milestone)
    assert main([str(tmp_path / "diag"), str(tmp_path / "plots")]) == 0
    index = json.loads((tmp_path / "diag/index.json").read_text())
    files = {entry["file"] for entry in index["plots"]}
    assert {entry["group"] for entry in index["plots"]} == {
        "data",
        "preferences",
        "reference",
        "policy",
        "ppo",
        "mcts",
        "milestones",
        "representation",
        "optimization",
        "memory",
        "losses",
    }
    assert "03_policy.png" in files and "09_milestones.png" in files
    assert any(
        panel["x_label"] == "Confidence bin" for panel in rendered["03_policy.png"]
    )
    assert any(
        panel["x_label"] == "Matched probability mass"
        for panel in rendered["08_mcts.png"]
    )
    assert any(panel["step"] for panel in rendered["09_milestones.png"])
    assert len(list((tmp_path / "plots").glob("*.png"))) == 12
    assert all(
        (tmp_path / "plots" / entry["file"]).is_file() for entry in index["plots"]
    )
