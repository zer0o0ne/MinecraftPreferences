"""src/mcpref/gui/textures.py against specs/gui-screens.md (Interface, Datasets): exactly the
files of T written by specs/shards.md (4), only the missing ones, members by exact name."""

import pytest
from store_disk import ancestors, snapshot
from textures_site import (
    DIRS,
    JAR,
    OTHER,
    TEXTURES,
    T,
    assert_failed,
    contents,
    jar,
    make_site,
    members,
    opened_for_writing,
    record_syncs,
    url,
)

from mcpref.gui.textures import main

PRESENT = b"present, not the member's bytes"
MISSING = [[p] for p in T] + [["gui/book.png", "gui/widgets.png"], list(T)]


@pytest.fixture
def site(http, catalogue, write_config, tmp_path, monkeypatch):
    return make_site(http, catalogue, write_config, tmp_path, monkeypatch)


def file_states(out) -> dict:
    """Snapshot of the regular files under out_dir: inode, mtime, bytes."""
    return {p: s for p, s in snapshot(out).items() if s[3] is not None}


def present(site, missing: list) -> dict:
    """Puts the files of T not in `missing` (bytes PRESENT); returns their states."""
    site.put({p: PRESENT for p in T if p not in missing})
    return file_states(site.out) if site.out.exists() else {}


def made_dirs(out) -> set:
    """make_dirs of the parent of every path of T: the parents and all their ancestors."""
    return set().union(*(ancestors(out / d) for d in DIRS))


def test_writes_exactly_t_byte_identical(site):
    """No other gui/ member, no jar, no .part left under out_dir."""
    code, server = site.run()
    assert code == 0
    assert contents(site.out) == DIRS | T
    assert [(r.method, r.path) for r in server.requests] == [("GET", "/client.jar")]


def test_jar_never_written_to_disk(site):
    """The only files opened for writing are the <path>.part of write_file."""
    config = site.config(url(site.serve(JAR)))
    with opened_for_writing() as written:
        assert main([config]) == 0
    assert set(written) == {site.out / f"{p}.part" for p in T}


def test_skips_without_request_when_t_present(site):
    """Nothing created, removed or rewritten under out_dir (directories included)."""
    present(site, [])
    before = snapshot(site.out)
    code, server = site.run()
    assert code == 0 and server.requests == []
    assert snapshot(site.out) == before


def test_rerun_after_complete_run_changes_nothing(site):
    assert site.run()[0] == 0
    before = snapshot(site.out)
    code, server = site.run()
    assert code == 0 and server.requests == []
    assert snapshot(site.out) == before


@pytest.mark.parametrize("missing", MISSING, ids="+".join)
def test_missing_files_downloaded_once_others_untouched(site, missing):
    before = present(site, missing)
    code, server = site.run()
    assert code == 0 and len(server.requests) == 1
    assert contents(site.out) == DIRS | T | dict.fromkeys(
        set(T) - set(missing), PRESENT
    )
    assert {p: file_states(site.out)[p] for p in before} == before


def test_files_outside_t_left_as_they_are(site):
    """Out of scope: foreign files kept; a leftover .part of a missing file is
    truncated and renamed by write_file (specs/shards.md (4))."""
    foreign = {"client.jar": b"jar", "gui/title/minecraft.png": b"title"}
    foreign |= {"gui/book.png": PRESENT, "gui/book.png.part": b"leftover of a present"}
    site.put(foreign)
    before = file_states(site.out)
    site.put({"gui/widgets.png.part": b"leftover, longer than the member's bytes"})
    assert site.run()[0] == 0
    assert contents(site.out) == DIRS | T | foreign | {"gui/title": None}
    assert {p: file_states(site.out)[p] for p in before} == before


@pytest.mark.parametrize(
    "absent, first",
    [(["gui/widgets.png", "font/ascii.png"], "font/ascii.png"),
     (["gui/widgets.png", "gui/book.png"], "gui/book.png")],
)  # fmt: skip
def test_absent_member_fails_naming_the_first_absent(site, capsys, absent, first):
    """Every missing member looked up before anything is written."""
    body = jar(members({p: d for p, d in T.items() if p not in absent}) + OTHER)
    code, server = site.run(body)
    err = capsys.readouterr().err
    assert_failed(code, err, site.out)
    assert first in err and all(p not in err for p in absent if p != first)
    assert len(server.requests) == 1


SIMILAR = {  # gui/book.png stored under a name other than its exact one
    "omitted": [],
    "other case": [(TEXTURES.upper() + "gui/book.png", T["gui/book.png"])],
    "under a prefix": [("x/" + TEXTURES + "gui/book.png", T["gui/book.png"])],
    "as a directory": [(TEXTURES + "gui/book.png/", b"")],
}


@pytest.mark.parametrize("similar", SIMILAR.values(), ids=SIMILAR.keys())
def test_member_found_by_exact_name_only(site, capsys, similar):
    rest = {p: d for p, d in T.items() if p != "gui/book.png"}
    code, _ = site.run(jar(members(rest) + similar + OTHER))
    err = capsys.readouterr().err
    assert_failed(code, err, site.out)
    assert "gui/book.png" in err


def test_absent_member_of_a_present_file_is_not_looked_up(site):
    site.put({"gui/book.png": PRESENT})
    rest = {p: d for p, d in T.items() if p != "gui/book.png"}
    assert site.run(jar(members(rest) + OTHER))[0] == 0
    assert contents(site.out) == DIRS | T | {"gui/book.png": PRESENT}


def test_last_duplicate_member_wins(site):
    """Library mapping: zipfile's exact lookup, the last duplicate wins."""
    body = jar(members(T) + [(TEXTURES + "gui/widgets.png", b"last")])
    assert site.run(body)[0] == 0
    assert contents(site.out) == DIRS | T | {"gui/widgets.png": b"last"}


@pytest.mark.parametrize("missing", [[], ["font/ascii.png", "gui/widgets.png"]])
def test_parents_synced_before_present_files_trusted(site, monkeypatch, missing):
    """Before the download only make_dirs runs; with nothing missing, nothing else."""
    present(site, missing)
    server = site.serve(JAR)
    log = record_syncs(monkeypatch, site.out, server)
    assert main([site.config(url(server))]) == 0
    assert {s.path for s in log if not s.requested} == made_dirs(site.out)
    assert all(s.read_only for s in log)


@pytest.mark.parametrize("missing", [list(T), ["font/ascii.png", "gui/widgets.png"]])
def test_missing_files_written_by_write_file(site, monkeypatch, missing):
    """specs/shards.md (4) per missing path p, in sorted order, after the download: make of
    its parent (parent and all ancestors synced), <p>.part synced, renamed, parent synced."""
    present(site, missing)
    server = site.serve(JAR)
    log = record_syncs(monkeypatch, site.out, server)
    assert main([site.config(url(server))]) == 0
    after = [s for s in log if s.requested]
    parts = [i for i, s in enumerate(after) if s.path.name.endswith(".part")]
    missing = sorted(missing)
    assert [after[i].path for i in parts] == [site.out / f"{p}.part" for p in missing]
    start = 0
    for p, i in zip(missing, parts, strict=True):
        made, part, parent = after[start:i], after[i], after[i + 1]
        assert {s.path for s in made} == ancestors((site.out / p).parent), p
        assert not any({p, f"{p}.part"} & s.entries for s in made), p
        assert f"{p}.part" in part.entries and p not in part.entries
        assert parent.path == (site.out / p).parent
        assert p in parent.entries and f"{p}.part" not in parent.entries
        start = i + 2
    assert start == len(after) and all(s.read_only for s in log)
