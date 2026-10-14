"""The runs of `main` of specs/gui-stack.md on the worlds of stack_world.py, shared by the
modules checking its outputs."""

import pytest
import torch

from .stack_world import NAN_WORLD, TIE_CONFIG, TIE_WORLD, run_main

# One intra-op thread per pytest-xdist worker: with torch's default (one per core) the
# workers oversubscribe the cores and a 300-step fit takes ~30x longer.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def nan_run(tmp_path_factory, repo):
    """The run of main on NAN_WORLD with CONFIG; read-only."""
    return run_main(tmp_path_factory.mktemp("nan"), repo, NAN_WORLD)


@pytest.fixture(scope="session")
def tie_run(tmp_path_factory, repo):
    """The run of main on TIE_WORLD with TIE_CONFIG; read-only."""
    return run_main(tmp_path_factory.mktemp("tie"), repo, TIE_WORLD, TIE_CONFIG)
