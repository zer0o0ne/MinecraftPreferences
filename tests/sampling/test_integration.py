"""Real Spark trajectory, readability and continuation of specs/env-sampling.md Tests."""

import dataclasses
import shutil
import statistics
import uuid

import numpy as np
import pytest
import torch

from mcpref.env.config import load_config as load_env_config
from mcpref.env.wrapper import Env
from mcpref.sampling.config import load_config
from mcpref.sampling.models import load_pretrained
from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import load_step


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_real_spark_short_trajectory_and_continuation(repo):
    configured = load_config(repo / "configs/env-sampling.yaml")
    cfg = dataclasses.replace(
        configured,
        out_dir=f"data/sampling-test/{uuid.uuid4().hex}",
        steps=1,
    )
    out = repo / cfg.out_dir
    env = Env(load_env_config(repo / cfg.env))
    try:
        modules = load_pretrained(cfg)
        sampler = Sampler(cfg, env, modules)
        sampler.reset()
        assert sampler.sample() == out / "shards/000000"
        first = load_step(out, 0)
        assert first.action >= 0
        assert first.probabilities.shape[0] > first.action
        torch.testing.assert_close(
            first.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
        )
        continued = Sampler(cfg, env, modules)
        continued.reset()
        assert continued.sample() == out / "shards/000001"
        second = load_step(out, 1)
        assert second.index == 1 and second.before.line == first.after.line
        np.testing.assert_array_equal(second.before.frame, first.after.frame)
        elapsed = [first.elapsed_ms, second.elapsed_ms]
        print(
            f"env sampling latency: median {statistics.median(elapsed):.3f} ms; "
            f"tail {max(elapsed):.3f} ms"
        )
    finally:
        env.close()
        shutil.rmtree(out, ignore_errors=True)
