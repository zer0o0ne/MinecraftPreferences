"""Fixed-model latent memoization against direct per-reference oracles."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mcpref.rl_pairs.history import History
from mcpref.rl_pairs.sample import (
    choose_seeds,
    compare,
    discover,
    insert_seeds,
    restore_work,
    seed_scores,
    select,
    write_work,
)
from mcpref.rl_pairs.scores import evaluating
from mcpref.rl_pairs.types import StateRef

from .helpers import Encoder, Preference, config, from_dict, vertex, work, write_history


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def uncached_select(states, history, encode, preference, alpha, cfg, rng, work_dir):
    scores_dir = work_dir.parent / "scores"
    seed_scores(states, encode, preference, scores_dir)
    seeds = choose_seeds(scores_dir, cfg.sample_states, rng)
    with evaluating(preference):
        insert_seeds(states, seeds, history, encode)
        completed, seen = restore_work(work_dir, history, rng)
        for ordinal in range(completed, len(seeds)):
            seed = states[seeds[ordinal]]
            observed = {}
            alternatives = discover(
                states, seeds, seeds[ordinal], history, alpha, cfg, rng, observed
            )
            comparisons = compare(
                seed, alternatives, seen, history, encode, preference, observed
            )
            write_work(
                work_dir, ordinal, seed, alternatives, comparisons, rng, observed
            )


def prepare(root: Path):
    cfg = config(
        root,
        sample_states=6,
        temporal_per_state=2,
        nearest_per_state=2,
        items_per_shard=1,
    )
    states = [StateRef("new", i) for i in range(6)]
    old = [StateRef("old", i) for i in range(2)]
    write_history(cfg, 0, [vertex(ref, [float(i), 0.0]) for i, ref in enumerate(old)])
    values = {
        ref: (float(i + 1), float(i % 2)) for i, ref in enumerate([*states, *old])
    }
    history = History.load(root / "history", cfg.history_per_shard, asdict(cfg))
    return cfg, states, old, values, history


def test_sampling_matches_direct_per_pair_oracle_and_encodes_each_ref_once(tmp_path):
    expected_root = tmp_path / "expected"
    cfg, states, old, values, history = prepare(expected_root)
    expected_encode = Encoder(values)
    expected_preference = Preference(default=(0.75, 0.1)).train()
    uncached_select(
        states,
        history,
        expected_encode,
        expected_preference,
        torch.tensor([1.0, 1.0]),
        cfg,
        np.random.default_rng(19),
        expected_root / "out/work",
    )

    actual_root = tmp_path / "actual"
    cfg, states, old, values, history = prepare(actual_root)
    actual_encode = Encoder(values)
    actual_preference = Preference(default=(0.75, 0.1)).train()
    select(
        states,
        history,
        actual_encode,
        actual_preference,
        torch.tensor([1.0, 1.0]),
        cfg,
        np.random.default_rng(19),
        actual_root / "out/work",
    )

    assert tree(actual_root / "out") == tree(expected_root / "out")
    assert actual_preference.calls == expected_preference.calls
    encoded = Counter(ref for call in actual_encode.calls for ref in call)
    assert encoded == Counter({ref: 1 for ref in [*states, *old]})
    assert sum(map(len, expected_encode.calls)) > len(encoded)


def test_publication_uses_fresh_final_model_latents(tmp_path):
    from mcpref.rl_pairs.cycle import publish_memory, sample_pairs

    cfg = config(tmp_path, sample_states=3, temporal_per_state=1)
    states = [StateRef("new", i) for i in range(3)]
    encode = Encoder({ref: (float(i + 1),) for i, ref in enumerate(states)})
    out = sample_pairs(
        cfg,
        0,
        states,
        encode,
        Preference(default=(0.8, 0.1)),
        torch.tensor([1.0]),
        np.random.default_rng(7),
    )
    before = {
        name: contents for name, contents in tree(out).items() if name != "run.json"
    }
    seeds = [from_dict(result["seed"]) for result in work(out)]

    encode.values = {ref: (100.0 + ref.index,) for ref in states}
    publish_memory(cfg, 0, encode)

    assert {
        name: contents for name, contents in tree(out).items() if name != "run.json"
    } == before
    history = tmp_path / "history/000000/shards"
    stored = {
        StateRef(value["episode"], value["index"]): value["key"]
        for path in history.glob("*/vertices.json")
        for value in json.loads(path.read_text())
    }
    assert {ref: stored[ref] for ref in seeds} == {
        ref: [100.0 + ref.index] for ref in seeds
    }
    encoded = Counter(ref for call in encode.calls for ref in call)
    assert encoded == Counter({ref: 2 for ref in states})
    assert encode.calls[-len(seeds) :] == [(ref,) for ref in seeds]
