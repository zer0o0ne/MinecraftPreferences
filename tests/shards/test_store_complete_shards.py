"""(2) of specs/shards.md: the number n of complete shards of D."""

import os

import pytest
from store_disk import populate, record_syncs, snapshot

from mcpref.shards.store import complete_shards


def test_complete_shards_of_missing_D_is_0(base, monkeypatch):
    log = record_syncs(monkeypatch)
    assert complete_shards(base / "D") == 0
    assert os.listdir(base) == [] and log == []


@pytest.mark.parametrize(
    "names, n",
    [
        ([], 0),
        (["000000.part/"], 0),
        (["000000/"], 1),
        (["000000/", "000001/", "000002/"], 3),
    ],
)
def test_complete_shards_counts(base, names, n):
    D = base / "D"
    populate(D, names)
    assert complete_shards(D) == n


def test_complete_shards_ignores_staged_shards_and_leaves_them(base):
    D = base / "D"
    names = ["000000/", "000001/", "000000.part/", "000002.part/", "000003.part"]
    populate(D, [*names, "1000000.part/"])
    populate(D / "000002.part", ["sub/", "y.bin"])
    before = snapshot(base)
    assert complete_shards(D) == 2
    assert snapshot(base) == before


def assert_value_error(D, message):
    """`complete_shards(D)` raises exactly ValueError(message), changing nothing on disk."""
    before = snapshot(D.parent)
    with pytest.raises(ValueError) as caught:
        complete_shards(D)
    assert type(caught.value) is ValueError and str(caught.value) == message
    assert snapshot(D.parent) == before


NON_CANONICAL = [
    "0000123",
    "12345",
    "-00001",
    "00000١",  # ARABIC-INDIC DIGIT ONE: int() reads it as 1
    "00_001",  # int() reads it as 1
    "+00001",
    " 00001",
    "000001 ",
    "0100000",
    "zzz",
    "12345.part",
    "0000123.part",
    ".part",
    "000000.part.part",
]


@pytest.mark.parametrize("name", NON_CANONICAL)
def test_complete_shards_non_canonical_name_is_foreign(base, name):
    D = base / "D"
    populate(D, ["000000/", "000001/", f"{name}/"])
    assert_value_error(D, f"{D}: foreign entry {name}")


def as_file(p):
    p.write_bytes(b"x")


def as_symlink_to_directory(p):
    p.symlink_to(p.parent / "000000", target_is_directory=True)


@pytest.mark.parametrize("make", [as_file, as_symlink_to_directory])
def test_complete_shards_non_directory_is_foreign(base, make):
    D = base / "D"
    populate(D, ["000000/", "000002/"])
    make(D / "000001")
    assert_value_error(D, f"{D}: foreign entry 000001")


@pytest.mark.parametrize(
    "names, smallest",
    [
        (["000000/", "9/", "12345/"], "12345"),  # as strings, not as numbers
        (["000000/", "a/", "B/"], "B"),  # by code point, not by locale
        (["000000/", "0000123/", "000002"], "000002"),
        (["000000.part/", "0000123/", "000001/"], "0000123"),
    ],
)
def test_complete_shards_names_smallest_foreign_entry(base, names, smallest):
    D = base / "D"
    populate(D, names)
    assert_value_error(D, f"{D}: foreign entry {smallest}")


@pytest.mark.parametrize(
    "names, j",
    [
        (["000001/"], "000000"),
        (["000000/", "000002/", "000005/"], "000001"),
        (["000000/", "000001/", "000003/", "000004/"], "000002"),
        (["000000/", "000001.part/", "000002/"], "000001"),
    ],
)
def test_complete_shards_names_smallest_missing_shard(base, names, j):
    D = base / "D"
    populate(D, names)
    assert_value_error(D, f"{D}: shard {j} missing")


@pytest.mark.parametrize(
    "names, foreign",
    [(["000001/", "zzz/"], "zzz"), (["000000/", "000003/", "1/"], "1")],
)
def test_complete_shards_foreign_entry_before_gap(base, names, foreign):
    D = base / "D"
    populate(D, names)
    assert_value_error(D, f"{D}: foreign entry {foreign}")


@pytest.mark.parametrize("names", [[], ["000000/", "000001/", "000002.part/"]])
def test_complete_shards_syncs_D(base, monkeypatch, names):
    D = base / "D"
    populate(D, names)
    log = record_syncs(monkeypatch)
    complete_shards(D)
    assert [s.path for s in log] == [D] and all(s.read_only for s in log)
