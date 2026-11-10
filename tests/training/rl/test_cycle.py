"""C1-C6: exact fake-CPU two-cycle scenario and artifact-based recovery."""

import hashlib
import json
import os

import pytest
import torch

import mcpref.training.rl.cycle as cycle_module
import mcpref.training.rl.learning.loop as loop_module
from mcpref.training.optim import update
from training.rl.cycle_world import Env, install, prepare, read, run
from training.rl.learning.loss_fakes import modules
from training.rl.pairs.helpers import vertices, work


def test_exact_two_cycle_roles_order_publish_and_stream_rebuild(
    tmp_path, repo, monkeypatch
):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    root, path = prepare(tmp_path, repo)
    events, batches, created, pair_rng, updates = [], [], [], [], []
    install(monkeypatch, events, batches)

    def builder(*args, **kwargs):
        value = modules()
        created.append(value)
        return value

    monkeypatch.setattr(cycle_module, "build_modules", builder)
    real_sample, real_publish = cycle_module.sample_pairs, cycle_module.publish_memory

    def sampled(cfg, cycle, states, encode, preference, alpha, rng):
        pair_rng.append((cycle, rng.bit_generator.state.copy()))
        events.append(("pairs-start", cycle, [ref.index for ref in states]))
        result = real_sample(cfg, cycle, states, encode, preference, alpha, rng)
        events.append(("pairs-end", cycle))
        return result

    def published(cfg, cycle, encode, **kwargs):
        assert (root / f"data/rl_cycle/{cycle:06d}/model.pt").exists()
        events.append(("publish", cycle))
        return real_publish(cfg, cycle, encode, **kwargs)

    monkeypatch.setattr(cycle_module, "sample_pairs", sampled)
    monkeypatch.setattr(cycle_module, "publish_memory", published)
    real_train = cycle_module.train

    def trained(cfg, modules, index_dir, initial_optimizer):
        cycle = int(index_dir.parent.name)
        assert not (root / f"data/rl_preference_history/{cycle:06d}").exists()
        events.append(("train-start", cycle))
        result = real_train(cfg, modules, index_dir, initial_optimizer)
        events.append(("train-end", cycle))
        return result

    monkeypatch.setattr(cycle_module, "train", trained)

    def counted(*args, **kwargs):
        updates.append(len(updates) + 1)
        return update(*args, **kwargs)

    monkeypatch.setattr(loop_module, "update", counted)
    env = Env(events)
    out0 = run(root, path, env)
    run0 = read(out0 / "run.json")
    index0 = read(out0 / "index/run.json")
    assert run0["complete"] and run0["phase"] == "finalized"
    assert index0["counts"] == {"fresh": 2, "pending": 0, "ppo": 1,
                                "preference": 1, "items": 3}  # fmt: skip
    assert read(out0 / "tail/shards/000000/items.json") == {
        "next_step": 2,
        "indices": [1],
    }
    assert events.index(("pairs-end", 0)) < events.index(("train-start", 0))
    assert events.index(("train-end", 0)) < events.index(("publish", 0))
    assert index0["batches"] == 2 and index0["steps"] == 2
    assert len(updates) == index0["batches"]
    final_scale = created[0]["phi"].scale.detach()
    seeds = work(root / "data/rl_preference_pairs/000000")
    stored = {
        entry["index"]: entry
        for entry in vertices(root / "data/rl_preference_history", 0)
    }
    for result in seeds:
        index = result["seed"]["index"]
        expected = [
            float(final_scale * (index + 1)),
            float(final_scale * (0.5 * index - 1)),
        ]
        assert stored[index]["key"] == pytest.approx(expected)
    initial_pair_run = read(root / "data/rl_preference_pairs/000000/run.json")
    out1 = run(root, path, env)
    index1 = read(out1 / "index/run.json")
    assert index1["counts"] == {"fresh": 2, "pending": 1, "ppo": 2,
                                "preference": index1["counts"]["preference"],
                                "items": index1["counts"]["items"]}  # fmt: skip
    assert read(out1 / "tail/shards/000000/items.json") == {
        "next_step": 4,
        "indices": [3],
    }
    records = [item for batch in batches for item in batch]
    cycle1_records = records[index0["counts"]["items"] :]
    assert {item["kind"] for item in cycle1_records} == {
        "fresh",
        "pending",
        "preference",
    }
    assert [item["index"] for item in cycle1_records if item["kind"] == "fresh"] == [
        2,
        3,
    ]
    assert [item["index"] for item in cycle1_records if item["kind"] == "pending"] == [
        1
    ]
    assert len(updates) == index0["batches"] + index1["batches"]
    assert pair_rng[1][1] == initial_pair_run["rng"]
    assert events[-1] == ("publish", 1)
    assert any(event[0] == "sampler-init" and event[1] == hashlib.sha256(
        (out0 / "model.pt").read_bytes()).hexdigest() for event in events)  # fmt: skip
    previous_comparisons = [c for result in work(root / "data/rl_preference_pairs/000001")
                            for c in result["comparisons"]]  # fmt: skip
    assert any(c["a"]["episode"].endswith("#000000") or c["b"]["episode"].endswith("#000000")
               for c in previous_comparisons)  # fmt: skip


def test_zero_preference_count_keeps_preference_groups(tmp_path, repo, monkeypatch):
    root, path = prepare(tmp_path, repo)
    initial_path = root / "data/pretrain-scaled-init/model.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    initial["phi"]["head.0.weight"].fill_(3.25)
    initial["phi"]["head.0.bias"].fill_(-1.75)
    torch.save(initial, initial_path)
    config = root / "configs/training/rl/preference-pairs.yaml"
    import yaml

    value = yaml.safe_load(config.read_text()) | {"sample_states": 0}
    config.write_text(yaml.safe_dump(value, sort_keys=False))
    pretrain = yaml.safe_load(
        (root / "configs/training/pretrain/config.yaml").read_text()
    )
    events, batches = [], []
    install(monkeypatch, events, batches)
    build_calls = []
    fake_builder = cycle_module.build_modules

    def builder(*args, **kwargs):
        build_calls.append((args, kwargs))
        return fake_builder(*args, **kwargs)

    monkeypatch.setattr(cycle_module, "build_modules", builder)
    out = run(root, path, Env(events))
    assert read(out / "index/run.json")["counts"]["preference"] == 0
    assert len(build_calls) == 1
    args, kwargs = build_calls[0]
    forwarded = kwargs.get("latent_init_scale", args[-1])
    assert forwarded == pretrain["latent_init_scale"]
    final = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    assert set(final) == {"phi", "pi", "P", "metra"}
    torch.testing.assert_close(
        final["phi"]["head.0.weight"],
        torch.full_like(final["phi"]["head.0.weight"], 3.25),
    )
    torch.testing.assert_close(
        final["phi"]["head.0.bias"],
        torch.full_like(final["phi"]["head.0.bias"], -1.75),
    )


def test_zero_ppo_count_writes_full_shift_metra_tail(tmp_path, repo, monkeypatch):
    root, path = prepare(tmp_path, repo)
    config = root / "configs/training/rl/ppo.yaml"
    import yaml

    value = yaml.safe_load(config.read_text()) | {"horizon": 3}
    config.write_text(yaml.safe_dump(value, sort_keys=False))
    events, batches = [], []
    install(monkeypatch, events, batches)
    out = run(root, path, Env(events))
    assert read(out / "index/run.json")["counts"]["ppo"] == 0
    assert read(out / "tail/shards/000000/items.json") == {
        "next_step": 2,
        "indices": [0, 1],
    }


class Interrupted(RuntimeError):
    pass


PHASES = ["sampled", "pairs", "indexed", "trained", "finalized"]


def interrupt_after(monkeypatch, phase, calls):
    names = {"sampled": "Sampler", "pairs": "sample_pairs", "indexed": "build_index",
             "trained": "train", "finalized": "publish_memory"}  # fmt: skip
    name = names[phase]
    real = getattr(cycle_module, name)
    if phase == "sampled":
        original_sample = real.sample

        def sample(self):
            original_sample(self)
            calls.append(phase)
            raise Interrupted(phase)

        monkeypatch.setattr(real, "sample", sample)
        return original_sample

    def wrapped(*args, **kwargs):
        real(*args, **kwargs)
        calls.append(phase)
        raise Interrupted(phase)

    monkeypatch.setattr(cycle_module, name, wrapped)
    return real


@pytest.mark.parametrize("phase", PHASES)
def test_recovery_reconciles_published_artifacts_without_duplicate_side_effects(
    tmp_path, repo, monkeypatch, capsys, phase
):
    root, path = prepare(tmp_path, repo)
    events, batches, calls = [], [], []
    install(monkeypatch, events, batches)
    real = interrupt_after(monkeypatch, phase, calls)
    env = Env(events)
    with pytest.raises(Interrupted):
        run(root, path, env)
    capsys.readouterr()
    if phase == "sampled":
        monkeypatch.setattr(cycle_module.Sampler, "sample", real)
    else:
        names = {"pairs": "sample_pairs", "indexed": "build_index", "trained": "train",
                 "finalized": "publish_memory"}  # fmt: skip
        monkeypatch.setattr(cycle_module, names[phase], real)
    out = run(root, path, env)
    captured = capsys.readouterr()
    assert calls == [phase]
    assert read(out / "run.json")["complete"]
    if phase == "indexed":
        assert "rl index: already done" in captured.err


def test_static_fingerprint_mismatch_aborts_unfinished_cycle(
    tmp_path, repo, monkeypatch
):
    root, path = prepare(tmp_path, repo)
    events, batches = [], []
    install(monkeypatch, events, batches)
    real = cycle_module.Sampler.sample

    def sampled(self):
        real(self)
        raise Interrupted

    monkeypatch.setattr(cycle_module.Sampler, "sample", sampled)
    env = Env(events)
    with pytest.raises(Interrupted):
        run(root, path, env)
    monkeypatch.setattr(cycle_module.Sampler, "sample", real)
    import yaml

    value = yaml.safe_load(path.read_text()) | {"seed": 1}
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(RuntimeError):
        run(root, path, env)


def test_run_fingerprints_cover_immutable_artifacts(tmp_path, repo, monkeypatch):
    root, path = prepare(tmp_path, repo)
    events, batches = [], []
    install(monkeypatch, events, batches)
    out = run(root, path, Env(events))
    value = read(out / "run.json")
    assert set(value["sha256"]) >= {
        "rl_config",
        "ppo",
        "rl_pairs",
        "initial_model",
        "trajectory",
        "pairs",
        "order",
        "index",
        "tail",
        "config",
        "pretrain_config",
        "env",
        "mcts",
        "perception_data",
        "perception",
        "action_mask",
        "policy",
        "preference",
        "engine",
        "classifier",
        "clips",
        "vpt_actions",
        "gui_screens",
    }
    pair_run = root / "data/rl_preference_pairs/000000/run.json"
    before = value["sha256"]["pairs"]
    pair_run.write_text(
        json.dumps(read(pair_run) | {"phase": "history", "complete": True})
    )
    assert read(out / "run.json")["sha256"]["pairs"] == before


def test_retrain_runs_after_joint_optimizer_and_before_memory_publication(
    tmp_path, repo, monkeypatch
):
    import yaml

    root, path = prepare(tmp_path, repo)
    path.write_text(
        yaml.safe_dump(
            yaml.safe_load(path.read_text()) | {"retrain": True}, sort_keys=False
        )
    )
    events, batches = [], []
    install(monkeypatch, events, batches)
    real_train = cycle_module.train

    def trained(cfg, modules, index_dir, initial_optimizer):
        result = real_train(cfg, modules, index_dir, initial_optimizer)
        events.append(("joint-end", 0))
        return result

    def separated(cfg, modules, index_dir, optimizer):
        assert isinstance(optimizer, torch.optim.AdamW)
        assert (index_dir.parent / "model.pt").exists()
        events.append(("retrain", 0))

    monkeypatch.setattr(cycle_module, "train", trained)
    monkeypatch.setattr(cycle_module, "retrain_heads", separated)
    monkeypatch.setattr(
        cycle_module,
        "publish_memory",
        lambda *args, **kwargs: events.append(("publish", 0)),
    )
    out = run(root, path, Env(events))
    assert read(out / "run.json")["phase"] == "finalized"
    assert events.index(("joint-end", 0)) < events.index(("retrain", 0))
    assert events.index(("retrain", 0)) < events.index(("publish", 0))
