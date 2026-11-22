"""Real sampler publication and changed-weight handover of continuous memory."""

import json
import shutil

import pytest
import torch

from mcpref.runtime.sampling.anchors import read_anchor
from mcpref.runtime.sampling.models import encode_stream
from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import read_observation
from runtime.sampling.continuous_world import modules, setup
from runtime.sampling.helpers import FakeEnv
from runtime.sampling.test_continuous import assert_tree_equal


def test_snapshots_precede_observation_and_cycle_endpoint_is_saved_once(
    site, monkeypatch
):
    setup(site, monkeypatch)
    sampler = Sampler(site.cfg, site.env, site.modules, replay_stride=2)
    sampler.reset()
    sampler.sample()
    assert read_anchor(site.out, 0, 2, site.modules.phi, "cpu", 2) is None
    anchor = read_anchor(site.out, 2, 2, site.modules.phi, "cpu", 2)
    assert anchor.layers[0].recurrent.item() == 3  # s_0 + s_1 only
    assert sampler.stream.state.layers[0].recurrent.item() == 6  # includes s_2
    old_anchor = (site.out / "shards/000000/anchors/000002.pt").read_bytes()
    retained = sampler.history.Y.clone()
    changed = modules("new-model", offset=10)
    resumed = Sampler(
        site.cfg, FakeEnv(), changed, replay_stride=2, handover_from="fake-model"
    )
    resumed.reset()
    resumed._prepare()
    torch.testing.assert_close(resumed.history.Y, retained, rtol=0, atol=0)
    torch.testing.assert_close(resumed.tree.base.Y, retained, rtol=0, atol=0)
    assert not changed.phi.events
    resumed.sample()
    assert [event[1] for event in changed.phi.events if event[0] == "stream"] == [3, 4]
    assert resumed.stream.state.layers[0].recurrent.item() == 15
    assert (site.out / "shards/000000/anchors/000002.pt").read_bytes() == old_anchor
    assert not (site.out / "shards/000001/anchors/000002.pt").exists()
    later = read_anchor(site.out, 4, 2, changed.phi, "cpu", 2)
    assert later.layers[0].recurrent.item() == 10
    observation, action = read_observation(site.out, 2)
    assert observation.line["id"] == 2 and action is not None


def test_changed_weight_continuation_matches_serialized_resume(site, monkeypatch):
    setup(site, monkeypatch)
    first = Sampler(site.cfg, site.env, site.modules, replay_stride=2)
    first.reset()
    first.sample()
    changed = Sampler(
        site.cfg,
        FakeEnv(),
        modules("updated", offset=7),
        replay_stride=2,
        handover_from="fake-model",
    )
    changed.reset()
    changed.sample()
    changed.sample()
    expected = encode_stream(changed.stream)
    expected_history = changed.history
    shutil.rmtree(site.out / "shards/000002")
    shutil.rmtree(site.root / "data/diagnostics")
    resumed = Sampler(
        site.cfg, FakeEnv(), modules("updated", offset=7), replay_stride=2
    )
    resumed.reset()
    resumed.sample()
    assert_tree_equal(encode_stream(resumed.stream), expected)
    assert_tree_equal(resumed.history.Y, expected_history.Y)
    assert_tree_equal(resumed.history.C, expected_history.C)


@pytest.mark.parametrize("change", ["wrong_handover", "stride", "missing_manifest"])
def test_incompatible_handover_fails_before_world_restore(site, monkeypatch, change):
    setup(site, monkeypatch)
    first = Sampler(site.cfg, site.env, site.modules, replay_stride=2)
    first.reset()
    first.sample()
    if change == "missing_manifest":
        (site.out / "shards/000000/anchors.json").unlink()
    env = FakeEnv()
    changed = Sampler(
        site.cfg,
        env,
        modules("updated"),
        replay_stride=3 if change == "stride" else 2,
        handover_from="wrong" if change == "wrong_handover" else "fake-model",
    )
    with pytest.raises(RuntimeError, match="memory anchors|model weights"):
        changed.reset()
    assert env.restores == env.resets == 0


def test_anchor_corruption_is_detected_before_replay(site, monkeypatch):
    setup(site, monkeypatch)
    first = Sampler(site.cfg, site.env, site.modules, replay_stride=2)
    first.reset()
    first.sample()
    path = site.out / "shards/000000/anchors/000002.pt"
    value = torch.load(path, weights_only=True)
    value["stream"]["layers"][0]["recurrent"].add_(1)
    torch.save(value, path)
    with pytest.raises(RuntimeError, match="changed continuous memory anchor"):
        read_anchor(site.out, 2, 2, site.modules.phi, "cpu", 2)
    manifest = json.loads((path.parent.parent / "anchors.json").read_text())
    assert set(manifest["files"]) == {"0", "2"}
