"""Reset, action records, histories, preferences and advance order of env-sampling.md."""

import json

import pytest
import torch
import yaml

from mcpref.runtime.mcts.tree import advance as real_advance
from mcpref.runtime.sampling.runner import Sampler
from mcpref.runtime.sampling.shards import load_state, load_step
from runtime.sampling.helpers import modules


def test_perception_history_caps_stream_and_rebuild_after_weight_update(site):
    for relative, field, value in (
        ("configs/models/perception.yaml", "history", 3),
        ("configs/datasets/perception-data.yaml", "states", 6),
    ):
        path = site.root / relative
        path.write_text(
            yaml.safe_dump(yaml.safe_load(path.read_text()) | {field: value})
        )
    sampler = Sampler(site.cfg, site.env, site.modules)
    assert sampler.states == sampler.stream.states == 3
    assert site.length_checks[-1][0][0] == 6
    sampler.reset()
    for _ in range(3):
        sampler.sample()
    assert sampler.stream.ids == [6]
    expected = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    torch.testing.assert_close(sampler.history.Y[0, :, 0], expected)
    updated = modules(fingerprint="updated", offset=10.0)
    restored = Sampler(site.cfg, site.env, updated)
    restored.reset()
    restored._prepare()
    assert restored.stream.ids == [6]
    torch.testing.assert_close(restored.history.Y[0, :, 0], expected + 10)


def test_explicit_reset_protocol_and_run_file_precedes_env_reset(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    with pytest.raises(RuntimeError):
        sampler.sample()
    seen = []
    reset = site.env.reset

    def checked_reset():
        seen.append(json.loads((site.out / "run.json").read_text()))
        return reset()

    site.env.reset = checked_reset
    sampler.reset()
    assert seen and seen[0]["config"] == site.mapping
    assert seen[0]["counts"] == {"trajectories": 0, "steps": 0}
    with pytest.raises(RuntimeError):
        sampler.reset()


def test_delayed_records_real_predicted_shifts_and_result_first_preferences(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    path = sampler.sample()
    assert path == site.out / "shards/000000"
    first, second = load_step(site.out, 0), load_step(site.out, 1)
    for index, record in enumerate((first, second)):
        assert record.index == index
        assert record.before.line["id"] == index
        assert record.after.line["id"] == index + 1
        assert record.probabilities.shape == (3,)
        assert record.probabilities.dtype == torch.float64
        torch.testing.assert_close(
            record.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        torch.testing.assert_close(record.delta, torch.ones(2, dtype=torch.float64))
        children = record.tree["root"]["children"]
        chosen = next(child for child in children if child["action"] == record.action)
        torch.testing.assert_close(
            record.predicted, chosen["x"] - record.tree["root"]["x"]
        )
    assert first.preference_close == {"succ": 1.0, "inc": 2.0}
    assert first.preference_start == {"succ": 1.0, "inc": 2.0}
    assert second.preference_close == {"succ": 1.0, "inc": 4.0}
    assert second.preference_start == {"succ": 2.0, "inc": 3.0}


def test_history_rows_dtype_cap_and_final_state(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    first, second = load_step(site.out, 0), load_step(site.out, 1)
    base0 = first.tree["base"]
    assert base0["Y"].shape == (1, 1, 2)
    assert base0["C"].shape == (1, 1, 4)
    assert base0["C"].dtype == base0["Y"].dtype == torch.float64
    torch.testing.assert_close(base0["C"][0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))  # fmt: skip
    base1 = second.tree["base"]
    torch.testing.assert_close(base1["Y"][0, :, 0], torch.tensor([0.0, 1.0], dtype=torch.float64))  # fmt: skip
    expected_c = torch.zeros(2, 4, dtype=torch.float64)
    expected_c[0, -1] = 1.0
    expected_c[1, first.action] = 1.0
    torch.testing.assert_close(base1["C"][0], expected_c)
    restored = load_state(site.out, site.modules.phi, 2, torch.device("cpu"))
    assert restored.next_step == 2 and restored.observation.line["id"] == 2
    torch.testing.assert_close(restored.history.Y[0, :, 0], torch.tensor([1.0, 2.0], dtype=torch.float64))  # fmt: skip
    final_c = torch.zeros(2, 4, dtype=torch.float64)
    final_c[0, first.action] = 1.0
    final_c[1, second.action] = 1.0
    torch.testing.assert_close(restored.history.C[0], final_c)


def test_ordinary_final_advance_and_successor_reuse_at_boundary(site, monkeypatch):
    advances = []

    def advance(tree, action, real, alpha, epsilon):
        advances.append((action, real.Y[0, -1, 0].item()))
        return real_advance(tree, action, real, alpha, epsilon)

    for name in (
        "mcpref.runtime.sampling.steps.advance",
        "mcpref.runtime.sampling.history.advance",
    ):
        monkeypatch.setattr(name, advance, raising=False)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    first, second = load_step(site.out, 0), load_step(site.out, 1)
    assert advances == [(first.action, 1.0), (second.action, 2.0)]
    advances.clear()
    site.events.clear()
    sampler.sample()
    third, fourth = load_step(site.out, 2), load_step(site.out, 3)
    assert (third.index, fourth.index) == (2, 3)
    assert (third.before.line["id"], fourth.before.line["id"]) == (2, 3)
    assert advances == [(third.action, 3.0), (fourth.action, 4.0)]
    assert third.preference_start == {"succ": 1.0, "inc": 6.0}
    assert fourth.preference_start == {"succ": 2.0, "inc": 7.0}
    streamed = [event[1] for event in site.events if event[0] == "stream"]
    assert streamed == [3, 4]
    assert (site.out / "shards/000001/steps/000000.pt").is_file()
    assert (site.out / "shards/000001/steps/000001.pt").is_file()


def test_all_stream_mcts_and_preference_calls_have_no_gradient(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    assert site.modules.phi.grad_modes
    assert site.modules.pi.grad_modes
    assert site.modules.P.grad_modes
    assert not any(site.modules.phi.grad_modes)
    assert not any(site.modules.pi.grad_modes)
    assert not any(site.modules.P.grad_modes)


def test_each_step_tree_is_an_independent_primitive_snapshot(site):
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    first, second = load_step(site.out, 0), load_step(site.out, 1)
    assert isinstance(first.tree, dict) and isinstance(second.tree, dict)
    assert first.tree is not second.tree
    assert first.tree["root"] is not second.tree["root"]
    first.tree["root"]["z"][0] = 99.0
    assert second.tree["root"]["z"][0].item() == 1.0


def test_alpha_comes_from_perception_config_in_phi_dtype_and_device(site, monkeypatch):
    from mcpref.runtime.mcts.search import decide as mcts_decide

    seen = []

    def decide(tree, mask, pi, P, alpha, cfg, rng, deadline):
        seen.append(alpha)
        return mcts_decide(tree, mask, pi, P, alpha, cfg, rng, deadline)

    monkeypatch.setattr("mcpref.runtime.sampling.steps.decide", decide)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    assert len(seen) == 2
    for alpha in seen:
        torch.testing.assert_close(alpha, torch.tensor([1.0, 2.0], dtype=torch.float64))
        assert alpha.dtype == site.modules.phi.weight.dtype
        assert alpha.device == site.modules.phi.weight.device
