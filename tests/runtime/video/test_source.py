"""Range endpoints, publication boundaries, and bounded record reads."""

import json
from pickle import UnpicklingError

import pytest
import torch

from mcpref.runtime.video.source import frames, snapshot
from runtime.video.conftest import write_trajectory


@pytest.mark.parametrize(
    "start,stop,values",
    [
        (0, 1, [0, 20]),
        (1, 3, [20, 40, 60]),
        (2, 5, [40, 60, 80, 100]),
        (3, 6, [60, 80, 100, 120]),
    ],
)
def test_endpoints_and_only_selected_records(source, monkeypatch, start, stop, values):
    real_load, loaded = torch.load, []

    def load(path, **kwargs):
        loaded.append(path)
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    stream = frames(snapshot(source), start, stop)
    assert loaded == []
    assert [int(frame[0, 0, 0]) for frame in stream] == values
    assert loaded == [
        source / f"shards/{i // 3:06d}/steps/{i % 3:06d}.pt" for i in range(start, stop)
    ]


def test_latest_is_fixed_while_new_shards_are_published(source):
    recording = snapshot(source)
    write_trajectory(source, 2, 3)
    assert recording.select(None, None, 2) == (4, 6)
    assert snapshot(source).select(None, None, 2) == (7, 9)
    assert [int(f[0, 0, 0]) for f in frames(recording, 4, 6)] == [80, 100, 120]


@pytest.mark.parametrize(
    "start,stop,last",
    [
        (None, None, None),
        (0, None, None),
        (None, 3, None),
        (-1, 2, None),
        (1, 1, None),
        (3, 2, None),
        (0, 7, None),
        (None, None, 0),
        (None, None, -1),
        (None, None, 7),
        (0, 3, 2),
        (None, 3, 2),
    ],
)
def test_invalid_selection(source, start, stop, last):
    with pytest.raises(ValueError):
        snapshot(source).select(start, stop, last)


@pytest.mark.parametrize(
    "change", ["index", "dtype", "shape", "odd", "missing", "corrupt"]
)
def test_bad_selected_record_fails(source, change):
    path = source / "shards/000001/steps/000001.pt"
    value = torch.load(path, weights_only=True)
    if change == "index":
        value["index"] = 99
    elif change == "dtype":
        value["after"]["frame"] = torch.zeros((24, 32, 3))
    elif change == "shape":
        value["after"]["frame"] = torch.zeros((12, 32, 3), dtype=torch.uint8)
    elif change == "odd":
        value["after"]["frame"] = torch.zeros((23, 32, 3), dtype=torch.uint8)
    elif change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    if change not in ("missing", "corrupt"):
        torch.save(value, path)
    error = {"missing": FileNotFoundError, "corrupt": UnpicklingError}.get(
        change, ValueError
    )
    with pytest.raises(error):
        list(frames(snapshot(source), 3, 6))


def test_no_completed_shards(tmp_path):
    (tmp_path / "shards/000000.part").mkdir(parents=True)
    with pytest.raises(ValueError, match="no completed"):
        snapshot(tmp_path)


def test_noncontiguous_first_trajectory(source):
    (source / "shards/000000/steps/000001.pt").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        snapshot(source)


def test_missing_final_step_is_not_a_shorter_trajectory(source):
    (source / "shards/000000/steps/000002.pt").unlink()
    with pytest.raises(ValueError, match="trajectory length"):
        snapshot(source)


def test_inherited_config_uses_frozen_length_and_stale_counts(source):
    (source / "run.json").write_text(
        json.dumps(
            {
                "config": {"extends": ["missing-config.yaml"]},
                "counts": {"trajectories": 1, "steps": 3},
            }
        )
    )
    assert snapshot(source).select(None, None, 2) == (4, 6)
    (source / "shards/000000/steps/000002.pt").unlink()
    with pytest.raises(ValueError, match="trajectory length"):
        snapshot(source)


def test_no_frozen_length_never_guesses(source):
    (source / "run.json").write_text(
        json.dumps(
            {
                "config": {"extends": ["missing-config.yaml"]},
                "counts": {"trajectories": 0, "steps": 0},
            }
        )
    )
    with pytest.raises(ValueError, match="retry after run.json"):
        snapshot(source)
