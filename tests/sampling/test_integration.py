"""Real Spark trajectory, readability and continuation of specs/env-sampling.md Tests."""

import dataclasses
import os
import shutil
import statistics
import uuid

import numpy as np
import pytest
import torch
from perception.world import STARTING as PERCEPTION_STARTING
from pretrain.index_run import repo_root
from runtime_configs import write_classifier_masks, write_repository

from mcpref.env.config import load_config as load_env_config
from mcpref.env.wrapper import Env
from mcpref.sampling.config import load_config
from mcpref.sampling.models import load_pretrained
from mcpref.sampling.runner import Sampler
from mcpref.sampling.shards import load_step


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_real_spark_short_trajectory_and_continuation(tmp_path, repo):
    root = tmp_path / "repo"
    write_repository(root, steps=1, states=12, alpha=PERCEPTION_STARTING["alpha"])
    write_classifier_masks(root)
    (root / "data").mkdir()
    os.symlink(
        repo / "data/pretrain",
        root / "data/pretrain-scaled-init",
        target_is_directory=True,
    )
    os.symlink(repo / "data/minestudio", root / "ms", target_is_directory=True)
    configured = load_config(root / "configs/env-sampling.yaml")
    cfg = dataclasses.replace(
        configured,
        out_dir=f"data/sampling-test/{uuid.uuid4().hex}",
        steps=1,
    )
    out = root / cfg.out_dir
    with repo_root(root):
        env = Env(load_env_config(root / cfg.env))
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
