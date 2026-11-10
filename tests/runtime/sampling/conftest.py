"""Shared fake-system fixture and one intra-op torch thread per xdist worker."""

import pytest
import torch
from training.pretrain.data.index_run import repo_root

from runtime.sampling.helpers import FakeStream, Site

torch.set_num_threads(1)


@pytest.fixture
def site(tmp_path, repo, monkeypatch):
    world = Site(tmp_path, repo)
    world.length_checks = []

    def check_lengths(*args, **kwargs):
        world.length_checks.append((args, kwargs))

    monkeypatch.setattr("mcpref.runtime.sampling.runner.PhiStream", FakeStream)
    monkeypatch.setattr("mcpref.runtime.sampling.models.PhiStream", FakeStream)
    monkeypatch.setattr("mcpref.runtime.sampling.runner.check_lengths", check_lengths)
    with repo_root(world.root):
        yield world
