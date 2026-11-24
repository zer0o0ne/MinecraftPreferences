"""Concurrent writers fail before mutations; process exit releases the same inode."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from infrastructure.runs.lock_process import attempt, holding
from mcpref.infrastructure.runs.lock import exclusive_run


def test_second_process_cannot_modify_existing_output(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"original checkpoint")
    (root / "metadata.json").write_bytes(b"original metadata")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with holding(root):
        result = attempt(root)
        assert result.returncode == 1
        assert "Another process is already using" in result.stderr
        assert str(root) in result.stderr
        assert "Traceback" not in result.stderr
        assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert attempt(root).returncode == 0
    assert (root / "checkpoint.pt").read_bytes() == b"new checkpoint"
    assert (root / "metadata.json").read_bytes() == before["metadata.json"]


def test_killed_process_releases_lock_without_deleting_file(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    lock = tmp_path / "run.lock"
    with holding(root) as process:
        inode = lock.stat().st_ino
        process.kill()
        process.wait(timeout=2)
        assert attempt(root).returncode == 0
        assert lock.stat().st_ino == inode


def test_second_writer_cannot_replace_an_inflight_checkpoint_shard(tmp_path):
    root = tmp_path / "run"
    staged = root / "checkpoints/000000.part"
    published = root / "checkpoints/000000"
    with holding(root, shard=True):
        assert (staged / "checkpoint.pt").read_bytes() == b"first weights"
        assert attempt(root, shard=True).returncode == 1
        assert (staged / "checkpoint.pt").read_bytes() == b"first weights"
        assert not published.exists()
    assert (published / "checkpoint.pt").read_bytes() == b"first weights"
    assert (published / "metadata.json").read_bytes() == b"first archive"


def test_nested_same_thread_keeps_outer_lock_and_directory_absent(tmp_path):
    root = tmp_path / "run"
    with exclusive_run(root):
        with exclusive_run(tmp_path / "unused" / ".." / "run"):
            assert attempt(root).returncode == 1
        assert attempt(root).returncode == 1
        assert not root.exists()
    with exclusive_run(root):
        assert not root.exists()


def test_another_thread_cannot_reenter_and_exception_releases(tmp_path):
    root = tmp_path / "run"

    def enter():
        with exclusive_run(root):
            pytest.fail("A different thread acquired an active run")

    with (
        pytest.raises(ValueError, match="interrupted"),
        exclusive_run(root),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        with pytest.raises(RuntimeError, match="already using"):
            pool.submit(enter).result(timeout=2)
        raise ValueError("interrupted")
    with exclusive_run(root):
        pass
