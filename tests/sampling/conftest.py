"""Shared fake-system fixture and one intra-op torch thread per xdist worker."""

import pytest
import torch
from pretrain.index_run import repo_root

from .helpers import FakeStream, Site

torch.set_num_threads(1)


@pytest.fixture
def site(tmp_path, repo, monkeypatch):
    world = Site(tmp_path, repo)
    world.length_checks = []

    def check_lengths(*args, **kwargs):
        world.length_checks.append((args, kwargs))

    monkeypatch.setattr("mcpref.sampling.runner.PhiStream", FakeStream)
    monkeypatch.setattr("mcpref.sampling.models.PhiStream", FakeStream)
    monkeypatch.setattr("mcpref.sampling.runner.check_lengths", check_lengths)
    with repo_root(world.root):
        yield world
