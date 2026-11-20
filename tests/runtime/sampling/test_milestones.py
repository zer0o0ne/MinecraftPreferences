"""Trajectory-local inventories and progress during diagnostic publication."""

from contextlib import contextmanager

from mcpref.diagnostics.core import runtime
from mcpref.diagnostics.core.artifacts import iter_records
from mcpref.runtime.sampling import runner
from mcpref.runtime.sampling.runner import Sampler


def test_repeated_sampling_keeps_milestones_local_to_each_trajectory(site):
    env_step = site.env.step

    def step(action):
        observation = env_step(action)
        observation.line["inventory"] = (
            [{"type": "diamond", "quantity": 1}] if site.env.index == 1 else []
        )
        return observation

    site.env.step = step
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
    sampler.sample()
    records = list(iter_records(sampler.diagnostics.out_dir, "milestones"))
    assert len(records) == 2
    assert [record["joined"]["first_times"]["diamonds"] for record in records] == [
        0,
        None,
    ]
    assert [record["joined"]["censored_at"] for record in records] == [1, 1]
    assert len(sampler._inventories) <= site.cfg.steps


def test_diagnostic_bars_are_active_during_publication(site, monkeypatch):
    active = {}
    original_bar = runner.diagnostic_bar
    original_write = runtime.write_record

    @contextmanager
    def bar(desc, total, initial, unit):
        with original_bar(desc, total, initial, unit) as progress:
            active[desc] = progress
            try:
                yield progress
                assert progress.n == total
            finally:
                del active[desc]

    def write(directory, kind, index, record):
        assert active[f"diagnostics {kind}"].n == index
        original_write(directory, kind, index, record)

    monkeypatch.setattr(runner, "diagnostic_bar", bar)
    monkeypatch.setattr(runtime, "write_record", write)
    sampler = Sampler(site.cfg, site.env, site.modules)
    sampler.reset()
    sampler.sample()
