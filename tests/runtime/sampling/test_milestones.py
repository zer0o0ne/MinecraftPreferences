"""Each sampling shard contributes only its own inventory observations."""

from mcpref.diagnostics.core.artifacts import iter_records
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
