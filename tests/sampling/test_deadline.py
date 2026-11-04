"""Whole-action soft deadline and elapsed-time boundaries of env-sampling.md (3)-(5)."""

import time

import pytest

from mcpref.mcts.search import decide as mcts_decide
from mcpref.mcts.tree import advance as real_advance
from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import load_step

from .helpers import FakeStream, Preference


def test_deadline_starts_before_perception_and_excludes_env_and_preferences(
    site, monkeypatch
):
    clock = [0.0]
    deadlines = []

    def perf_counter():
        return clock[0]

    stream_step = FakeStream.step

    def timed_stream(self, frame, line):
        clock[0] += 0.005
        return stream_step(self, frame, line)

    def decide(tree, mask, pi, P, alpha, cfg, rng, deadline):
        deadlines.append(deadline)
        clock[0] += 0.010
        return mcts_decide(tree, mask, pi, P, alpha, cfg, rng, deadline)

    def advance(tree, action, real, alpha, epsilon):
        clock[0] += 0.010
        return real_advance(tree, action, real, alpha, epsilon)

    preference_call = Preference.__call__

    def timed_preference(self, Ya, Yb):
        result = preference_call(self, Ya, Yb)
        clock[0] += 0.100
        return result

    env_step = site.env.step

    def timed_env(action):
        result = env_step(action)
        clock[0] += 0.200
        return result

    monkeypatch.setattr(time, "perf_counter", perf_counter)
    for name in (
        "mcpref.sampling.steps.perf_counter",
        "mcpref.sampling.steps.time.perf_counter",
    ):
        monkeypatch.setattr(name, perf_counter, raising=False)
    monkeypatch.setattr("mcpref.mcts.search.time.perf_counter", perf_counter)
    monkeypatch.setattr(FakeStream, "step", timed_stream)
    monkeypatch.setattr("mcpref.sampling.steps.decide", decide)
    for name in ("mcpref.sampling.steps.advance", "mcpref.sampling.history.advance"):
        monkeypatch.setattr(name, advance, raising=False)
    monkeypatch.setattr(Preference, "__call__", timed_preference)
    site.env.step = timed_env
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    first, second = load_step(site.out, 0), load_step(site.out, 1)
    # delta=5 gives 250 ms; safety_limit=25 ms leaves a 225 ms action budget.
    assert deadlines == pytest.approx([0.225, 0.440])
    assert first.elapsed_ms == pytest.approx(15.0)
    assert second.elapsed_ms == pytest.approx(25.0)
