"""Interface crash boundaries: byte-identical continuation without repeated network calls."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import tqdm
from progress_bars import Bar, states

from mcpref.rl_pairs.cycle import sample_pairs
from mcpref.rl_pairs.types import StateRef

from .helpers import Encoder, Preference, config


class Interrupted(RuntimeError):
    pass


def interrupt_bar(monkeypatch, desc, at):
    real = tqdm.tqdm.update
    raised = False

    def update(self, n=1):
        nonlocal raised
        result = real(self, n)
        if self.desc == desc and self.n == at and not raised:
            raised = True
            raise Interrupted(desc)
        return result

    monkeypatch.setattr(tqdm.tqdm, "update", update)


def data_tree(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and p.name != "run.json"}  # fmt: skip


def run(root, **changes):
    values = {
        "sample_states": 3,
        "temporal_per_state": 2,
        "items_per_shard": 1,
    } | changes
    cfg = config(root, **values)
    episode = [StateRef("e", i) for i in range(4)]
    encode = Encoder({r: (float(i + 1),) for i, r in enumerate(episode)})
    preference = Preference(default=(0.75, 0.0)).train()
    rng = np.random.default_rng(77)
    return cfg, episode, encode, preference, rng


def complete(root):
    cfg, episode, encode, preference, rng = run(root)
    out = sample_pairs(cfg, 0, episode, encode, preference, torch.tensor([1.0]), rng)
    return out, len(preference.calls)


@pytest.mark.parametrize(
    "desc,at,kept",
    [
        ("rl seed scoring", 1, "out/000000/scores/000000/scores.json"),
        ("rl pair sampling", 1, "out/000000/work/000000/result.json"),
        ("rl pair correction", 1, "out/000000/shards/000000/items.json"),
    ],
    ids=["score-shard", "seed-work-shard", "correction-shard"],
)
def test_interrupted_shard_continuation(tmp_path, monkeypatch, capsys, desc, at, kept):
    expected_out, expected_calls = complete(tmp_path / "baseline")
    expected = data_tree(tmp_path / "baseline")
    root = tmp_path / "resumed"
    cfg, episode, encode, preference, rng = run(root)
    interrupt_bar(monkeypatch, desc, at)
    with pytest.raises(Interrupted):
        sample_pairs(cfg, 0, episode, encode, preference, torch.tensor([1.0]), rng)
    capsys.readouterr()
    assert (root / kept).read_bytes() == expected[kept]
    calls_before = len(preference.calls)
    checkpoint = json.loads((root / "out/000000/run.json").read_text())
    rng.bit_generator.state = checkpoint["rng"]
    out = sample_pairs(cfg, 0, episode, encode, preference, torch.tensor([1.0]), rng)
    captured = capsys.readouterr()
    assert out == root / "out/000000"
    assert data_tree(root) == expected
    assert len(preference.calls) == expected_calls
    assert states(captured.err, desc)[0] == Bar(desc, at, {
        "rl seed scoring": 3, "rl pair sampling": 3, "rl pair correction": 3,
    }[desc], {"rl seed scoring": "state", "rl pair sampling": "seed",
              "rl pair correction": "pair"}[desc])  # fmt: skip
    if desc == "rl pair correction":
        assert len(preference.calls) == calls_before
    assert preference.training and preference.dummy.grad is None
    assert expected_out == tmp_path / "baseline/out/000000"


def prepare_history_cycle(root):
    cfg, _, _, _, _ = run(root, sample_states=1, temporal_per_state=0, capacity=1)
    old = [StateRef("old", 0)]
    sample_pairs(
        cfg,
        0,
        old,
        Encoder({old[0]: (0.0,)}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(4),
    )


def history_cycle(root, interrupt=False, monkeypatch=None):
    cfg = config(root, sample_states=2, temporal_per_state=1, capacity=1,
                 items_per_shard=1)  # fmt: skip
    episode = [StateRef("new", i) for i in range(2)]
    encode = Encoder({r: (float(i + 1),) for i, r in enumerate(episode)})
    preference = Preference(default=(0.8, 0.0)).train()
    if interrupt:
        real = shutil.rmtree
        raised = False

        def remove(path, *args, **kwargs):
            nonlocal raised
            if Path(path) == root / "history/000000" and not raised:
                raised = True
                raise Interrupted("history")
            return real(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", remove)
    return sample_pairs(
        cfg,
        1,
        episode,
        encode,
        preference,
        torch.tensor([1.0]),
        np.random.default_rng(8),
    ), preference


def test_interruption_after_history_generation_has_no_replayed_side_effects(
    tmp_path, monkeypatch, capsys
):
    prepare_history_cycle(tmp_path / "baseline")
    prepare_history_cycle(tmp_path / "resumed")
    expected_out, _ = history_cycle(tmp_path / "baseline")
    expected = data_tree(tmp_path / "baseline")
    with pytest.raises(Interrupted):
        history_cycle(tmp_path / "resumed", True, monkeypatch)
    capsys.readouterr()
    new_run = tmp_path / "resumed/history/000001/run.json"
    assert new_run.exists() and (tmp_path / "resumed/history/000000").exists()
    out, preference = history_cycle(tmp_path / "resumed")
    captured = capsys.readouterr()
    assert out == Path(str(expected_out).replace("/baseline/", "/resumed/"))
    assert data_tree(tmp_path / "resumed") == expected
    assert preference.calls == []
    assert all("already done" in line for line in captured.err.splitlines())
