"""Interface progress lines, completion markers, and config/input fingerprints."""

import dataclasses
import json

import numpy as np
import pytest
import torch
from progress_bars import Bar, assert_progress, screen, states

from mcpref.rl_pairs.cycle import sample_pairs
from mcpref.rl_pairs.types import StateRef

from .helpers import Encoder, Preference, config, items, shards, work


def episode():
    return [StateRef("e", i) for i in range(3)]


def encode(states):
    return Encoder({r: (float(i + 1),) for i, r in enumerate(states)})


def test_all_progress_bars_and_initial_positions(tmp_path, capsys):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1, capacity=1)
    states_in = episode()
    out = sample_pairs(
        cfg,
        0,
        states_in,
        encode(states_in),
        Preference(default=(0.8, 0.0)),
        torch.tensor([1.0]),
        np.random.default_rng(2),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert_progress(captured.err)
    expected = [Bar("rl seed scoring", 2, 2, "state"),
                Bar("rl pair sampling", 2, 2, "seed"),
                Bar("rl pair correction", 1, 1, "pair"),
                Bar("rl pair cleanup", 1, 1, "vertex")]  # fmt: skip
    assert screen(captured.err) == expected
    for bar in expected:
        assert states(captured.err, bar.desc)[0] == Bar(
            bar.desc, 0, bar.total, bar.unit
        )
    assert len(items(out)) == 1


def test_empty_input_has_all_nothing_to_do_lines(tmp_path, capsys):
    cfg = config(tmp_path, sample_states=0)
    out = sample_pairs(
        cfg,
        0,
        [],
        Encoder({}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert screen(captured.err) == [
        "rl seed scoring: nothing to do",
        "rl pair sampling: nothing to do",
        "rl pair correction: nothing to do",
        "rl pair cleanup: nothing to do",
    ]
    assert items(out) == []


def test_complete_rerun_is_byte_stable_already_done_and_has_no_network_calls(
    tmp_path, capsys
):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1, capacity=1)
    states_in = episode()
    encoder = encode(states_in)
    preference = Preference(default=(0.8, 0.0))
    sample_pairs(
        cfg,
        0,
        states_in,
        encoder,
        preference,
        torch.tensor([1.0]),
        np.random.default_rng(2),
    )
    capsys.readouterr()
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls = len(preference.calls)
    sample_pairs(
        cfg,
        0,
        states_in,
        encoder,
        preference,
        torch.tensor([1.0]),
        np.random.default_rng(999),
    )
    captured = capsys.readouterr()
    assert len(preference.calls) == calls
    assert {p: p.read_bytes() for p in before} == before
    assert screen(captured.err) == [
        "rl seed scoring: already done",
        "rl pair sampling: already done",
        "rl pair correction: already done",
        "rl pair cleanup: already done",
    ]


def test_completion_markers_fingerprints_and_shard_layout(tmp_path):
    cfg = config(tmp_path, sample_states=3, temporal_per_state=2, items_per_shard=1)
    states_in = episode()
    out = sample_pairs(
        cfg,
        0,
        states_in,
        encode(states_in),
        Preference(default=(0.8, 0.0)),
        torch.tensor([1.0]),
        np.random.default_rng(2),
    )
    run = json.loads((out / "run.json").read_text())
    assert run["complete"] and run["phase"] == "history" and run["pairs"] == 3
    assert set(run["sha256"]) == {"config", "input"}
    assert json.loads((tmp_path / "history/000000/run.json").read_text())["complete"]
    assert len(shards(out / "scores", "scores.json")) == 3
    assert len(work(out)) == len(shards(out / "shards", "items.json")) == 3


def test_config_fingerprint_mismatch_aborts(tmp_path):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1)
    states_in = episode()
    sample_pairs(
        cfg,
        0,
        states_in,
        encode(states_in),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(2),
    )
    changed = dataclasses.replace(cfg, capacity=cfg.capacity - 1)
    with pytest.raises(RuntimeError):
        sample_pairs(
            changed,
            0,
            states_in,
            encode(states_in),
            Preference(),
            torch.tensor([1.0]),
            np.random.default_rng(2),
        )


def test_input_fingerprint_mismatch_aborts(tmp_path):
    cfg = config(tmp_path, sample_states=2, temporal_per_state=1)
    states_in = episode()
    sample_pairs(
        cfg,
        0,
        states_in,
        encode(states_in),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(2),
    )
    changed = [StateRef("e", 9), *states_in[1:]]
    with pytest.raises(RuntimeError):
        sample_pairs(
            cfg,
            0,
            changed,
            encode(changed),
            Preference(),
            torch.tensor([1.0]),
            np.random.default_rng(2),
        )
