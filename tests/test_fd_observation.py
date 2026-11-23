"""Descriptor observers must preserve the durability tests' real OS checks."""

import os

import pytest
from fd_observation import fd_path, is_read_only


@pytest.mark.parametrize("mode", [os.O_RDONLY, os.O_WRONLY, os.O_RDWR])
def test_observe_file_access_and_close(tmp_path, mode):
    path = tmp_path.resolve() / "file"
    path.write_bytes(b"payload")
    fd = os.open(path, mode)
    try:
        assert fd_path(fd) == path
        assert is_read_only(fd) == (mode == os.O_RDONLY)
        renamed = path.with_name("renamed")
        path.rename(renamed)
        assert fd_path(fd) == renamed
    finally:
        os.close(fd)
    assert fd_path(fd) is None


def test_observe_directory(tmp_path):
    path = tmp_path.resolve()
    fd = os.open(path, os.O_RDONLY)
    try:
        assert fd_path(fd) == path
        assert is_read_only(fd)
    finally:
        os.close(fd)
