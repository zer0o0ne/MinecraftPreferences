"""Real shared index/reader, diagnostic transport and a small trainable model."""

import json
from types import SimpleNamespace

import torch
from diagnostics.core.world import write_config
from training.pretrain.continuous.data.test_training import model, observation_step
from training.pretrain.continuous.data.world import make_world
from training.pretrain.continuous.engine_world import config, context

from mcpref.datasets.continuous.recovery import seal
from mcpref.datasets.continuous.timeline import identity
from mcpref.diagnostics.core import artifacts, runtime, state
from mcpref.diagnostics.producers import preference as producer
from mcpref.infrastructure.runs.files import write_json
from mcpref.models.preference.network import Preference
from mcpref.training.pretrain.continuous import steps
from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index
from mcpref.training.pretrain.continuous.diagnostics import (
    hooks,
    records,
    reference_replay,
    reference_store,
    selection,
)
from mcpref.training.pretrain.continuous.schedule import epoch_updates


def setup(tmp_path, monkeypatch, *, preferences=True):
    world = make_world(tmp_path, monkeypatch, K=3)
    if preferences:
        validation_pairs(world, tmp_path)
    else:
        (world.root / "preference-map/000000/pairs.jsonl").write_text("")
    index.run(world.cfg, world.path)
    data = Index(world.out)
    for module in (
        artifacts,
        runtime,
        state,
        producer,
        hooks,
        records,
        reference_store,
        selection,
    ):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(selection, "load_config", lambda _: None)
    monkeypatch.setattr(selection, "load_masks", lambda _: world.masks)
    monkeypatch.setattr(steps, "observation_step", observation_step)
    monkeypatch.setattr(reference_replay, "observation_step", observation_step)
    monkeypatch.setenv("MCPREF_PROFILE", "")
    return world, data


def validation_pairs(world, root):
    directory = world.root / "fragments/000002"
    maps = []
    for alias, ids in (("validation-a", ["f2:0", "f2:1"]), ("validation-b", ["f2:1"])):
        key = identity(ids)
        clip = directory / "clips" / key
        write_json(
            clip / "clip.json",
            {
                "id": key,
                "dataset": "dataset",
                "fragment_id": "f2",
                "observation_ids": ids,
                "frames": [world.positions[v] for v in ids],
                "classical_valid": True,
            },
        )
        maps.append(
            {
                "status": "mapped",
                "endpoint": alias,
                "original_id": alias,
                "clip_ref": str(clip.relative_to(root)),
                "canonical_history_id": key,
            }
        )
    (directory / "mappings.jsonl").write_text(
        "".join(json.dumps(v) + "\n" for v in maps)
    )
    (directory / "state.json").unlink()
    seal(directory)
    path = world.root / "preference-map/000000/pairs.jsonl"
    pairs = [json.loads(line) for line in path.read_text().splitlines()]
    # This A spans two source episodes; its truncated history starts in the second.
    pairs[3].update(a="c", b="a")
    path.write_text("".join(json.dumps(row) + "\n" for row in pairs))
    with path.open("a") as stream:
        for i, labels in enumerate(({"inc": 0, "p": 0.5}, {"inc": 1, "p": None})):
            stream.write(
                json.dumps(
                    {
                        "id": f"validation:{i}",
                        "a": "validation-a",
                        "b": "validation-b",
                        "labels": labels,
                    }
                )
                + "\n"
            )


def training(tmp_path, world, data, name, *, gradients=False):
    cfg_path = write_config(
        tmp_path / f"{name}-diagnostics.yaml",
        out_dir=f"{name}-diagnostics",
        bootstrap_samples=4,
    )
    cfg = config(
        epochs=1,
        tbptt_states=3,
        batch_size=3,
        checkpoint_steps=2,
        diagnostic_window=1,
        diagnostics=cfg_path.name,
        reference_steps=2,
        reference_pairs=2,
        calibration_bins=3,
        action_copy_eps=0.1,
        saturation_threshold=0.01,
        inc_classification_threshold=0.5,
        quantiles=(0.5,),
        gradient_diagnostics=gradients,
        action_mask="mask.yaml",
    )
    modules = model(world)
    if data.run["populations"]["train"]["N_P"]:
        cfg.groups["P_no_decay"] = {"lr": 0.01, "weight_decay": 0}
        modules["P"] = Preference(
            SimpleNamespace(
                history=2,
                width=4,
                heads=1,
                hidden=8,
                layers=("S",),
                rope_base=10000,
                norm_eps=1e-5,
                seed=17,
            ),
            2,
        )
    else:
        cfg.groups.pop("P")
        for key in ("P", "P_loss"):
            modules.pop(key)
    total = epoch_updates(data, cfg) * cfg.epochs
    result = context(tmp_path / name, cfg, modules, data, total)
    result.root.mkdir(exist_ok=True)
    return result


def weights(path):
    return torch.load(path / "model.pt", weights_only=True)


def equal_weights(first, second):
    assert first.keys() == second.keys()
    for module in first:
        if not isinstance(first[module], dict):
            assert first[module] == second[module]
        else:
            for key, value in first[module].items():
                torch.testing.assert_close(value, second[module][key], rtol=0, atol=0)


def accepted(root):
    index = json.loads((root / "index.json").read_text())
    result = {}
    for item in index["records"]:
        path = root / item["kind"] / f"{item['index']:06d}" / "metrics.json"
        record = json.loads(path.read_text())
        assert record["id"] not in result
        result[record["id"]] = record
    return result
