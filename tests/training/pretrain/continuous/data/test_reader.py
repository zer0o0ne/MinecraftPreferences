"""Chronological suffix resume, target lookahead and exact population event oracles."""

from types import SimpleNamespace

import pytest
from training.pretrain.continuous.data.world import make_world

from mcpref.training.pretrain.continuous.data import index
from mcpref.training.pretrain.continuous.data.reader import Index, StreamReader
from mcpref.training.pretrain.continuous.schedule import epoch_updates


def test_reader_keeps_grid_gaps_idle_history_origin_and_boundary_targets(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    index.run(world.cfg, world.path)
    data = Index(world.out)
    stream = StreamReader(data, world.phi_cfg, None)
    try:
        fragment = next(data.fragments("train"))
        items = list(stream.observations(fragment))
        assert [item.grid_index for item in items] == [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
        assert [item.index for item in items] == list(range(10))
        assert [item.grid_index for item in items if item.interval["valid"]] == [
            0,
            1,
            6,
            7,
            9,
        ]
        assert items[2].interval["known"] and items[2].interval["actions"] == []
        assert not items[3].interval["known"]
        assert not items[4].interval["known"]
        assert items[7].interval["actions"] == ["turn_up"]
        assert items[7].interval["legal_actions"] == []
        assert items[0].interval["a"] == "f0:0"
        assert items[0].interval["b"] == "f0:1"
        assert items[0].shard != items[2].shard
        assert not items[-1].interval["valid"]
        assert items[-1].meta == "10.0,64.0,0.0"
        assert all(item.origin == items[0].line for item in items)
        suffix = list(stream.observations(fragment, start=5))
        assert [(v.id, v.meta, v.interval) for v in suffix] == [
            (v.id, v.meta, v.interval) for v in items[5:]
        ]
        assert suffix[0].origin == items[0].line
    finally:
        data.close()


def test_lookahead_does_not_request_future_frames_and_resume_does_not_read_old_frames(
    tmp_path, monkeypatch
):
    world = make_world(tmp_path, monkeypatch)
    index.run(world.cfg, world.path)
    data = Index(world.out)
    stream = StreamReader(data, world.phi_cfg, None)
    calls, original = [], stream.payload.frame

    def frame(shard, position):
        calls.append([shard, position])
        return original(shard, position)

    monkeypatch.setattr(stream.payload, "frame", frame)
    try:
        fragment = next(data.fragments("train"))
        iterator = stream.observations(fragment, start=2)
        first = next(iterator)
        assert calls == [world.positions[first.id]]
        assert first.id == "f0:2"
        iterator.close()
        with pytest.raises(ValueError, match="cursor"):
            next(stream.observations(fragment, start=11))
    finally:
        data.close()


@pytest.mark.parametrize("B,updates", [(1, 13), (2, 7), (3, 5), (20, 1)])
def test_update_count_crosses_fragments_and_ignores_gradient_horizon(
    tmp_path, monkeypatch, B, updates
):
    world = make_world(tmp_path, monkeypatch, K=4)
    run = index.run(world.cfg, world.path)
    data = Index(world.out)
    try:
        assert epoch_updates(data, SimpleNamespace(batch_size=B)) == updates
    finally:
        data.close()
    assert run["populations"]["train"]["N_phi"] == 7
    assert run["populations"]["train"]["N_pi"] == 7
