"""Fixtures of tests of specs/shards.md."""

from pathlib import Path

import pytest


@pytest.fixture
def base(tmp_path) -> Path:
    """A fresh directory, its path free of symlinks (as `/proc/self/fd` reports it)."""
    return tmp_path.resolve()
