"""`patch` of specs/engine.md (5) on the synthetic jars of jars.py against the oracle there:
order, $R$ absent, $A$ present once, repeated names skipped (the first written wins), data
equal and recompressed, the kept `ZipInfo` fields, `flag_bits` 0, `external_attr`."""

import zipfile
import zlib
from functools import cache

import pytest

from mcpref.engine.patch import patch

from .jars import KEPT, SHARED, expected, lwjgl_jars, read_entries, removed, source_jar

# The entries of jars.py's S in R, by hand, except `org/lwjgl/` (also in A).
R_OF_S = {"liblwjgl.so", "liblwjgl.so.sha1", "lwjgl32.dll.git", "libglfw_wayland.so",
          "OpenAL.dll", "libjemalloc.dylib.sha1", "org/lwjgl/Version.class",
          "org/lwjgl/system/Library.class", "META-INF/versions/10/org/lwjgl/",
          "META-INF/versions/9/org/lwjgl/system/StackWalkUtil.class"}  # fmt: skip


@cache
def inputs() -> tuple[bytes, list]:
    return source_jar(), lwjgl_jars()


@pytest.fixture
def patched(tmp_path) -> tuple[list, list]:
    """(the entries of the patched jar, the oracle's entries)."""
    source, jars = inputs()
    (tmp_path / "S.jar").write_bytes(source)
    patch(tmp_path / "S.jar", jars, tmp_path / "J.jar")
    return read_entries(tmp_path / "J.jar"), expected(source, jars)


def deflated_size(data: bytes) -> int:
    """zipfile's deflate at the default level (`compresslevel` None)."""
    c = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    return len(c.compress(data) + c.flush())


def test_order_and_names(patched):
    """S's entries not in R in their order, then A in jar order, each name once."""
    got, want = patched
    names = [info.filename for info, _ in got]
    assert names == [info.filename for info, _ in want]
    assert len(names) == len(set(names))


def test_removed_absent_added_present(patched):
    """S's entries in R absent (`org/lwjgl/` comes back from A: test below); the names of
    A present; a jar's names outside A (its manifest, index, directories above A, a native
    in a classes jar, a class in a natives jar) absent."""
    names = {info.filename for info, _ in patched[0]}
    assert not R_OF_S & names
    assert all(removed(n) for n in R_OF_S)  # the hand list is R's
    assert {
        "org/lwjgl/",
        "META-INF/versions/9/org/lwjgl/",
        "linux/arm64/",
        "org/lwjgl/lwjgl_tinyfd/Cls.class",
        "linux/arm64/org/lwjgl/lwjgl_tinyfd/liblwjgl_tinyfd.so.sha1",
    } <= names
    assert not [n for n in names if n.endswith(("InNatives.class", "InClasses.so"))]
    outside = {
        "META-INF/versions/",
        "META-INF/versions/9/",
        "linux/",
        "META-INF/INDEX.LIST",
    }
    assert not outside & names


def test_repeated_names_keep_the_first(patched):
    """A repeated directory comes from the first jar holding it (dated 2000 + jar index);
    a name of S kept in the jar keeps S's entry."""
    got = {info.filename: (info, data) for info, data in patched[0]}
    assert got["org/lwjgl/"][0].date_time[0] == 2000  # classes of lwjgl
    assert got["META-INF/versions/9/org/lwjgl/"][0].date_time[0] == 2000
    assert got["linux/arm64/org/lwjgl/"][0].date_time[0] == 2001  # natives of lwjgl
    assert got[SHARED][1] == b"from S"


def test_data_equal(patched):
    got, want = patched
    assert [data for _, data in got] == [data for _, data in want]


def test_kept_fields(patched):
    """filename, date_time, compress_type, comment, extra, create_system, create_version,
    extract_version, reserved, internal_attr of the entry's own source."""
    got, want = patched
    fields = lambda info: {k: getattr(info, k) for k in KEPT}
    assert [fields(i) for i, _ in got] == [fields(i) for i, _ in want]


def test_volume_written_as_zero(patched):
    """Some sources carry a non-zero `volume` (S's and a jar's); writestr writes 0."""
    got, want = patched
    assert {i.volume for i, _ in want} >= {0, 1, 2, 3}
    assert {i.volume for i, _ in got} == {0}


def test_flag_bits_cleared(patched):
    """The sources carry the UTF-8 flag 0x800; writestr sets `flag_bits` to 0."""
    got, want = patched
    assert {i.flag_bits for i, _ in want} == {0x800}
    assert {i.flag_bits for i, _ in got} == {0}


def test_external_attr(patched):
    """A non-zero `external_attr` kept; 0 becomes 0o600 << 16."""
    got, want = patched
    assert {i.external_attr for i, _ in want} >= {0, (0o40755 << 16) | 0x10}
    expected_attrs = [i.external_attr or 0o600 << 16 for i, _ in want]
    assert [i.external_attr for i, _ in got] == expected_attrs


def test_recompressed_at_default_level(patched, tmp_path):
    """Each entry's data compressed again by its compression type at the default level
    (the sources are deflated at level 1); sizes and CRCs recomputed; the zip is sound."""
    got, want = patched
    for info, data in got:
        assert (info.file_size, info.CRC) == (len(data), zlib.crc32(data))
        if info.compress_type == zipfile.ZIP_STORED:
            assert info.compress_size == len(data)
        else:
            assert info.compress_size == deflated_size(data)
    sizes = [
        (src.compress_size, deflated_size(data))
        for src, data in want
        if src.compress_type == zipfile.ZIP_DEFLATED and data
    ]
    assert any(a != b for a, b in sizes)  # the sources are not at the default level
    with zipfile.ZipFile(tmp_path / "J.jar") as z:
        assert z.testzip() is None


def test_leftover_output_overwritten(tmp_path):
    """An existing file at `out` is replaced by the patched jar alone."""
    source, jars = inputs()
    (tmp_path / "S.jar").write_bytes(source)
    with zipfile.ZipFile(tmp_path / "J.jar", "w") as z:
        z.writestr("leftover.txt", b"x" * 100_000)
    patch(tmp_path / "S.jar", jars, tmp_path / "J.jar")
    names = [i.filename for i, _ in read_entries(tmp_path / "J.jar")]
    assert names == [i.filename for i, _ in expected(source, jars)]
