"""C4 and equations (4)-(5): schedule, shared updates, checkpoints, and exact resume."""

import numpy as np
import pytest
import torch
import tqdm
from pretrain.index_run import repo_root
from progress_bars import Bar, screen, states
from training.schedule_oracle import epoch

from mcpref.rl.loop import train

from .loop_world import (
    checkpoints,
    install_steps,
    losses,
    model,
    run_train,
    write_world,
)
from .loss_fakes import modules


def test_schedule_item_order_one_shared_update_and_full_postfix(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_world(tmp_path, repo)
    calls, updates, postfix = [], [], []
    install_steps(monkeypatch, calls, updates)
    real = tqdm.tqdm.set_postfix

    def record(self, ordered_dict=None, refresh=True, **kwargs):
        if self.desc == "rl cycle":
            postfix.append({**(ordered_dict or {}), **kwargs})
        return real(self, ordered_dict, refresh, **kwargs)

    monkeypatch.setattr(tqdm.tqdm, "set_postfix", record)
    err, out = run_train(root, cfg, index, modules())
    assert out == "" and updates == [1, 2, 3, 4, 5, 6]
    rng = np.random.Generator(np.random.PCG64([cfg.seed, 3]))
    source = [[{"kind": "fresh", "index": 0}, {"kind": "preference", "index": 10},
               {"kind": "fresh", "index": 1}],
              [{"kind": "pending", "index": 2}, {"kind": "preference", "index": 11}]]  # fmt: skip
    expected = []
    for _ in range(2):
        for shard, positions in epoch(rng, [3, 2], 2):
            expected.extend(source[shard][int(position)] for position in positions)
    assert [item for batch, _ in calls for item in batch] == expected
    entries = [entry for shard in losses(index) for entry in shard]
    assert len(entries) == 6 and [entry["step"] for entry in entries] == list(
        range(1, 7)
    )
    assert postfix == [{key: entry[key] for key in ["ppo", "shift", "metra", "P", "loss"]
                        if entry[key] is not None} for entry in entries]  # fmt: skip
    assert list(entries[0]["lr"]) == list(cfg.groups)
    final = model(index)
    assert set(final) == {"phi", "pi", "P", "metra"}
    assert all(
        tensor.device.type == "cpu"
        for state in final.values()
        for tensor in state.values()
    )
    drawn = states(err, "rl cycle")
    assert drawn[0] == Bar("rl cycle", 0, 6, "step")
    assert drawn[-1].n == 6 and drawn[-1].postfix is not None


def test_none_gradients_leave_other_modules_and_dual_untouched(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_world(tmp_path, repo)
    mods = modules()
    before = {name: {key: value.clone() for key, value in module.state_dict().items()}
              for name, module in mods.items() if name != "phi"}  # fmt: skip
    install_steps(monkeypatch, [])
    run_train(root, cfg, index, mods)
    for name, state in before.items():
        for key, value in state.items():
            assert value.equal(mods[name].state_dict()[key])
    assert not mods["phi"].scale.detach().equal(torch.tensor(1.2, dtype=torch.float64))


class Interrupted(RuntimeError):
    pass


def interrupt_at(monkeypatch, at):
    real = tqdm.tqdm.update
    raised = False

    def update(self, n=1):
        nonlocal raised
        result = real(self, n)
        if self.desc == "rl cycle" and self.n == at and not raised:
            raised = True
            raise Interrupted
        return result

    monkeypatch.setattr(tqdm.tqdm, "update", update)


def assert_same_model(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].keys() == b[name].keys()
        for key in a[name]:
            assert a[name][key].equal(b[name][key])


@pytest.mark.parametrize("at", [2, 3], ids=["inside-epoch", "epoch-boundary"])
def test_interrupted_training_restores_rng_schedule_and_modules_bitwise(
    tmp_path, repo, monkeypatch, capsys, at
):
    whole, cfg, index = write_world(tmp_path / "whole", repo)
    install_steps(monkeypatch, [])
    run_train(whole, cfg, index, modules())
    expected_model = model(index)
    expected_losses = losses(index)
    cut, cfg, index = write_world(tmp_path / "cut", repo)
    install_steps(monkeypatch, [])
    interrupt_at(monkeypatch, at)
    with pytest.raises(Interrupted), repo_root(cut):
        train(cfg, modules(), index, None)
    capsys.readouterr()
    torch.manual_seed(9182)
    torch.rand(4)
    calls = []
    install_steps(monkeypatch, calls)
    err, _ = run_train(cut, cfg, index, modules())
    assert_same_model(model(index), expected_model)
    assert losses(index) == expected_losses
    assert len(calls) == 6 - at
    assert states(err, "rl cycle")[0] == Bar("rl cycle", at, 6, "step")


def test_checkpoint_retention_keeps_every_loss_log(tmp_path, repo, monkeypatch):
    root, cfg, index = write_world(tmp_path, repo, keep_checkpoints=2)
    install_steps(monkeypatch, [])
    run_train(root, cfg, index, modules())
    assert [path.parent.name for path in checkpoints(index)] == ["000004", "000005"]
    assert len(losses(index)) == 6
    assert [[entry["step"] for entry in shard] for shard in losses(index)] == [
        [i] for i in range(1, 7)
    ]


def test_restore_final_step_performs_no_updates_and_prints_already_done(
    tmp_path, repo, monkeypatch
):
    root, cfg, index = write_world(tmp_path, repo)
    install_steps(monkeypatch, [])
    run_train(root, cfg, index, modules())
    (index.parent / "model.pt").unlink()
    calls = []
    install_steps(monkeypatch, calls)
    err, out = run_train(root, cfg, index, modules())
    assert calls == [] and out == ""
    assert screen(err) == ["rl cycle: already done"]
    assert (index.parent / "model.pt").exists()


def test_fresh_cycle_resets_torch_seed(tmp_path, repo, monkeypatch):
    root, cfg, index = write_world(tmp_path, repo, epochs=1)
    torch.manual_seed(991)
    torch.rand(5)
    calls = []
    install_steps(monkeypatch, calls)
    run_train(root, cfg, index, modules())
    expected = []
    torch.manual_seed(cfg.seed)
    for _ in range(3):
        expected.append(float(torch.rand((), dtype=torch.float64)))
    assert [draw for _, draw in calls] == expected


def test_repeated_epochs_recompute_histories_but_keep_behavior_and_labels(
    tmp_path, repo, monkeypatch
):

    from .helpers import write_json
    from .loss_fakes import BEHAVIOR, PreferenceLoss, install, load_step
    from .test_loss import fresh, preference

    root, cfg, index = write_world(tmp_path, repo)
    for path in (index / "shards").glob("*/items.json"):
        path.unlink()
    items = [fresh(0), preference()]
    write_json(index / "shards/000000/items.json", {"items": items})
    (index / "shards/000001").rmdir()
    write_json(
        index / "run.json",
        {
            "complete": True,
            "counts": {"fresh": 1, "pending": 0, "ppo": 1, "preference": 1, "items": 2},
            "batches": 1,
            "steps": 2,
            "sha256": "fake",
        },
    )
    events = []
    install(monkeypatch, events)
    behavior, labels = [], []

    def recorded_step(out_dir, index):
        record = load_step(out_dir, index)
        behavior.append(record.probabilities.clone())
        return record

    monkeypatch.setattr("mcpref.rl.loss.load_step", recorded_step)
    real_loss = PreferenceLoss.forward

    def recorded_loss(self, out, supplied):
        labels.append([dict(label) for label in supplied])
        return real_loss(self, out, supplied)

    monkeypatch.setattr(PreferenceLoss, "forward", recorded_loss)
    run_train(root, cfg, index, modules(events))
    assert len(behavior) == len(labels) == 2
    assert all(value.equal(BEHAVIOR) for value in behavior)
    assert labels == [[{"p": 0.75, "inc": 0}]] * 2
    assert len([event for event in events if event[0] == "P"]) == 6
    assert len([event for event in events if event[0] == "encode"]) >= 8
