"""Equation (10): cleanup scores, ties, repeated removal, and symmetric deletion."""

import numpy as np
import pytest
import torch

from mcpref.rl_pairs.cycle import sample_pairs
from mcpref.rl_pairs.types import StateRef

from .helpers import (
    Encoder,
    Preference,
    comparison,
    config,
    from_dict,
    vertex,
    vertices,
    write_history,
)


def linked_records(refs):
    a, b, c, d = refs
    return [
        vertex(
            a,
            [0.0],
            comparisons=[comparison(a, b, 0.8, 0.1, 1), comparison(a, c, 0.7, 0.2, 4)],
        ),
        vertex(b, [1.0], comparisons=[comparison(b, a, 0.2, 0.1, 1)]),
        vertex(c, [2.0], comparisons=[comparison(c, a, 0.3, 0.2, 4)]),
        vertex(d, [3.0]),
    ]


def test_cleanup_repeated_removal_reference_deletion_and_count_subtraction(tmp_path):
    refs = [StateRef("old", i) for i in range(4)]
    a, _, c, _ = refs
    cfg = config(tmp_path, sample_states=0, capacity=2)
    write_history(cfg, 0, linked_records(refs))
    sample_pairs(
        cfg,
        1,
        [],
        Encoder({}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    stored = {from_dict(v): v for v in vertices(tmp_path / "history", 1)}
    assert set(stored) == {a, c}
    assert stored[a]["comparisons"] == [comparison(a, c, 0.7, 0.2, 4)]
    assert stored[c]["comparisons"] == [comparison(c, a, 0.3, 0.2, 4)]
    assert stored[a]["unused"] == stored[c]["unused"] == 1


@pytest.mark.parametrize("unused", [5, 6], ids=["stable-id-tie", "larger-unused"])
def test_cleanup_score_tie_uses_larger_unused_then_stable_id(tmp_path, unused):
    refs = [StateRef("old", i) for i in range(4)]
    records = [vertex(refs[0], [0.0], comparisons=[comparison(refs[0], refs[1], 0.5, 0.0, 2)]),
               vertex(refs[1], [1.0], comparisons=[comparison(refs[1], refs[0], 0.5, 0.0, 2)]),
               vertex(refs[2], [2.0], unused=unused), vertex(refs[3], [3.0], unused=5)]  # fmt: skip
    cfg = config(tmp_path, sample_states=0, capacity=3)
    write_history(cfg, 0, records)
    sample_pairs(
        cfg,
        1,
        [],
        Encoder({}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    assert {from_dict(v) for v in vertices(tmp_path / "history", 1)} == {
        refs[0],
        refs[1],
        refs[3],
    }


def test_exact_capacity_boundary_removes_nothing(tmp_path):
    refs = [StateRef("old", i) for i in range(2)]
    cfg = config(tmp_path, sample_states=0, capacity=2)
    write_history(cfg, 0, [vertex(refs[0], [0.0], unused=2),
                           vertex(refs[1], [1.0], unused=4)])  # fmt: skip
    sample_pairs(
        cfg,
        1,
        [],
        Encoder({}),
        Preference(),
        torch.tensor([1.0]),
        np.random.default_rng(0),
    )
    stored = vertices(tmp_path / "history", 1)
    assert [from_dict(v) for v in stored] == refs
    assert [v["unused"] for v in stored] == [3, 5]
