"""Whole-history shift decisions from explicit kept-scale examples."""

import pytest

from mcpref.datasets.continuous.mapping import map_clip


def clip(states):
    return {
        "path": "old/a",
        "sha256": "hash",
        "kind": "preference",
        "clip": {"id": "a", "relpath": "source", "states": states},
    }


def timeline(n, missing=(), cut=()):
    kept = [t for t in range(n) if t not in cut]
    return {
        "id": "stream",
        "world_group": "world",
        "kappa": kept,
        "observations": {
            p: f"s:{p // 6}" for p in range(0, len(kept), 6) if p not in missing
        },
    }


@pytest.mark.parametrize(
    "offset,shift", [(0, 0), (1, -1), (2, -2), (3, -3), (4, 2), (5, 1)]
)
def test_one_common_shift(offset, shift):
    result = map_clip(
        clip([6 + offset, 12 + offset, 18 + offset]), timeline(40), 0, 6, 3
    )
    assert result["shift"] == shift
    assert result["observations"] == [
        f"s:{(6 + offset + shift) // 6 + i}" for i in range(3)
    ]


def test_tie_uses_later_grid_if_earlier_missing():
    result = map_clip(clip([3, 9, 15]), timeline(25, missing={0}), 0, 6, 3)
    assert result["shift"] == 3
    assert result["observations"] == ["s:1", "s:2", "s:3"]


def test_fragment_edge_is_not_independently_rounded():
    assert (
        map_clip(clip([4, 10, 16]), timeline(18), 0, 6, 3)["status"]
        == "insufficient_fragment_extent"
    )


def test_shift_limit_uses_kept_steps_not_raw_ticks():
    result = map_clip(clip([10, 16, 22]), timeline(30, cut=range(1, 10)), 0, 6, 3)
    assert result["shift"] == -1
    assert result["observations"] == ["s:0", "s:1", "s:2"]


def test_nonuniform_history_cannot_be_rounded_state_by_state():
    assert (
        map_clip(clip([2, 8, 14]), timeline(25, cut={4}), 0, 6, 3)["status"]
        == "nonuniform_kept_history"
    )
