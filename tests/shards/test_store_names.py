"""Shard names name(k) of specs/shards.md: `shard_path`, and (1), (2) at k = 10**6."""

import os
from pathlib import Path

import pytest
from store_disk import contents, populate, snapshot

from mcpref.shards.store import complete_shards, shard_path, write_shard

NAMES = [
    (0, "000000"),
    (7, "000007"),
    (123, "000123"),
    (999999, "999999"),
    (10**6, "1000000"),
    (1234567, "1234567"),
]


@pytest.mark.parametrize("k, name", NAMES)
def test_shard_path(k, name):
    assert shard_path(Path("/data/D"), k) == Path("/data/D") / name


def test_shard_path_of_relative_D():
    assert shard_path(Path("out/D"), 12) == Path("out/D/000012")


def test_write_shard_at_10_6_after_999999(base):
    D = base / "D"
    populate(D, ["999999/"])
    with write_shard(D, 10**6) as staged:
        assert staged == D / "1000000.part"
        (staged / "f.bin").write_bytes(b"f")
    assert sorted(os.listdir(D)) == ["1000000", "999999"]
    assert contents(D / "1000000") == {"f.bin": b"f"}


@pytest.mark.parametrize(
    "names, exception, message",
    [
        (["999998/"], ValueError, "{D}: shard 999999 missing before 1000000"),
        (["999999/", "1000000/"], FileExistsError, "{D}/1000000 exists"),
    ],
)
def test_write_shard_checks_at_10_6(base, names, exception, message):
    D = base / "D"
    populate(D, names)
    before = snapshot(base)
    with pytest.raises(exception) as caught, write_shard(D, 10**6):
        pass
    assert type(caught.value) is exception
    assert str(caught.value) == message.format(D=D)
    assert snapshot(base) == before


def test_complete_shards_of_999999_and_1000000_misses_000000(base):
    D = base / "D"
    populate(D, ["999999/", "1000000/"])
    with pytest.raises(ValueError) as caught:
        complete_shards(D)
    assert str(caught.value) == f"{D}: shard 000000 missing"
