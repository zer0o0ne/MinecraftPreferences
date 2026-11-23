"""Fixtures of tests of specs/shards.md."""

from pathlib import Path

import pytest


@pytest.fixture
def base(tmp_path) -> Path:
    """A fresh directory with the canonical path reported by descriptor inspection."""
    return tmp_path.resolve()
