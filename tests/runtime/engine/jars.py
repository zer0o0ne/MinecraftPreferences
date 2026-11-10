"""Synthetic jars of specs/engine.md (5): a source jar S varying every `ZipInfo` field (5)
keeps, the LWJGL classes and natives jars of the modules M, and an oracle of the patched jar
transcribed from Notation ($R$, $A$ with the repeat rule) and (5). Every entry carries the
UTF-8 flag 0x800, as in the jars of [E] and Maven; some have `external_attr` 0, some a
non-zero `volume` (both patched into the central headers: zipfile writes neither)."""

import io
import random
import re
import struct
import zipfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

MODULES = ("lwjgl", "lwjgl-opengl", "lwjgl-glfw", "lwjgl-stb", "lwjgl-openal",
           "lwjgl-jemalloc", "lwjgl-tinyfd")  # fmt: skip
KEPT = ("filename", "date_time", "compress_type", "comment", "extra", "create_system",
        "create_version", "extract_version", "reserved", "internal_attr")  # fmt: skip
CAFE = b"\xfe\xca\x00\x00"  # the jar tool's extra field: tag 0xCAFE, no data
NATIVE = re.compile(
    r"(lib)?(lwjgl(_opengl|_stb|_tinyfd)?|glfw(_wayland)?|openal|OpenAL|jemalloc)(32)?"
    r"\.(so|dll|dylib)(\.git|\.sha1)?"
)
VERSIONED = re.compile(r"META-INF/versions/[0-9]+/org/lwjgl/")
WORDS = ("org", "lwjgl", "glfw", "class", "void", "static", "final", "return", "native",
         "long", "int", "buffer", "memory", "stack", "pointer", "window")  # fmt: skip
SHARED = "linux/arm64/org/lwjgl/README"  # kept in S and in the first natives jar


def text(seed: int, words: int = 400) -> bytes:
    """Compressible bytes whose deflate sizes differ between levels 1 and 6."""
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words)).encode()


def at(year: int, second: int = 0) -> dict:
    return {"date_time": (year, 3, 4, 5, 6, second)}


def source_entries() -> list:
    """S: (name, data, ZipInfo fields, external_attr 0), in order; `# R`: in R."""
    dirs, stored = (0o40755 << 16) | 0x10, {"compress_type": ZIP_STORED}
    # fmt: off
    return [
        ("META-INF/", b"", {**stored, "extra": CAFE, **at(2020)}, True),
        ("META-INF/MANIFEST.MF", text(1), {"comment": b"manifest", **at(2020, 2)}, True),
        ("Start.class", text(2), {"create_system": 3, "external_attr": 0o100644 << 16, "volume": 1}, False),
        ("liblwjgl.so", random.Random(3).randbytes(300), stored, True),  # R
        ("liblwjgl.so.sha1", b"0123abcd", {**at(2020, 4)}, True),  # R
        ("lwjgl32.dll.git", b"git", {}, True),  # R
        ("libglfw_wayland.so", b"elf", {"create_system": 3}, False),  # R
        ("OpenAL.dll", b"pe", {}, True),  # R
        ("libjemalloc.dylib.sha1", b"sha", {}, True),  # R
        ("libjinput-linux64.so", text(4), {**stored, "internal_attr": 1}, True),
        ("libfliteWrapper.so", text(5), {"create_version": 45, **at(2021)}, True),
        ("org/", b"", {**stored, "external_attr": dirs, "create_system": 3}, False),
        ("org/lwjgl/", b"", stored, True),  # R
        ("org/lwjgl/Version.class", text(6), {}, True),  # R
        ("org/lwjgl/system/Library.class", text(7), {}, True),  # R
        ("net/minecraft/client/Main.class", text(8), {"reserved": 1, **at(2022)}, True),
        ("META-INF/versions/9/org/lwjgl/system/StackWalkUtil.class", text(9), {}, True),  # R
        ("META-INF/versions/10/org/lwjgl/", b"", stored, True),  # R
        ("META-INF/versions/9/module-info.class", text(10), {"extract_version": 10}, True),
        ("x/org/lwjgl/Shadow.class", text(11), {"comment": b"not at the start"}, True),
        (SHARED, b"from S", {**at(2023)}, True),
        ("pack.png", random.Random(12).randbytes(500), {**stored, "extract_version": 10, "volume": 2}, True),
    ]
    # fmt: on


def jar_entries(i: int, module: str, natives: bool) -> list:
    """Jar i of the 2|M| jars (classes or natives of `module`), dated in year 2000 + i."""
    pkg, d = module.replace("-", "_"), at(2000 + i)
    meta = [("META-INF/", b"", {"extra": CAFE, **d}, False),
            ("META-INF/MANIFEST.MF", f"Name: {module}".encode(), d, False)]  # fmt: skip
    tail = [("META-INF/versions/9/module-info.class", text(100 + i), d, False)]
    if natives:
        lib = f"linux/arm64/org/lwjgl/{pkg}/lib{pkg}.so"
        body = [("linux/", b"", d, False), ("linux/arm64/", b"", d, False),
                ("linux/arm64/org/", b"", d, False), ("linux/arm64/org/lwjgl/", b"", d, True),
                (f"linux/arm64/org/lwjgl/{pkg}/", b"", d, False),
                (lib, random.Random(i).randbytes(200), {"compress_type": ZIP_STORED, **d}, False),
                (lib + ".sha1", text(200 + i, 8), {"comment": b"sha1", **d}, True),
                (f"org/lwjgl/{pkg}/InNatives.class", b"natives only give linux/arm64/", d, False)]  # fmt: skip
        body += [(SHARED, b"from the natives", d, False)] if i == 1 else []
    else:
        body = [("META-INF/versions/", b"", d, False), ("META-INF/versions/9/", b"", d, False),
                ("META-INF/versions/9/org/lwjgl/", b"", d, False),
                (f"META-INF/versions/9/org/lwjgl/{pkg}/Multi.class", text(300 + i), d, True),
                ("org/", b"", d, False), ("org/lwjgl/", b"", d, False),
                (f"org/lwjgl/{pkg}/", b"", {"extra": CAFE, **d}, False),
                (f"org/lwjgl/{pkg}/Cls.class", text(400 + i), {"create_system": 3, "volume": 3, **d}, False),
                (f"linux/arm64/org/lwjgl/{pkg}/InClasses.so", b"classes give no natives", d, False)]  # fmt: skip
    return meta + body + tail


def build(entries: list) -> bytes:
    """A zip of `entries`; deflated data at level 1 (not the default level of (5))."""
    buf = io.BytesIO()
    with ZipFile(buf, "w") as z:
        for name, data, fields, _ in entries:
            info = ZipInfo(name, (2019, 1, 2, 3, 4, 6))
            info.compress_type = ZIP_DEFLATED
            for key, value in fields.items():
                setattr(info, key, value)
            z.writestr(info, data, compresslevel=1)
    zero = {name for name, _, _, zero_attr in entries if zero_attr}
    volumes = {name: fields.get("volume", 0) for name, _, fields, _ in entries}
    return flagged(buf.getvalue(), zero, volumes)


def flagged(data: bytes, zero: set, volumes: dict) -> bytes:
    """`data` with the flag 0x800 in every local and central header, `external_attr` 0 in
    the central headers of the names in `zero` and each name's `volumes` (its disk number
    start)."""
    buf, z = bytearray(data), ZipFile(io.BytesIO(data))
    p = z.start_dir
    for _ in z.infolist():
        n, m, k = struct.unpack_from("<3H", buf, p + 28)
        (offset,) = struct.unpack_from("<L", buf, p + 42)
        for flag in (p + 8, offset + 6):
            struct.pack_into(
                "<H", buf, flag, struct.unpack_from("<H", buf, flag)[0] | 0x800
            )
        name = buf[p + 46 : p + 46 + n].decode()
        struct.pack_into("<H", buf, p + 34, volumes[name])
        if name in zero:
            struct.pack_into("<L", buf, p + 38, 0)
        p += 46 + n + m + k
    return bytes(buf)


def source_jar() -> bytes:
    return build(source_entries())


def lwjgl_jars() -> list[tuple[bytes, bool]]:
    """The 2|M| jars in order: for each module, classes then natives."""
    order = [(m, natives) for m in MODULES for natives in (False, True)]
    return [(build(jar_entries(i, m, n)), n) for i, (m, n) in enumerate(order)]


def removed(name: str) -> bool:
    """$R$ of Notation."""
    root = "/" not in name and NATIVE.fullmatch(name)
    return bool(name.startswith("org/lwjgl/") or VERSIONED.match(name) or root)


def added(name: str, natives: bool) -> bool:
    """$A$ of Notation without the repeat rule."""
    if natives:
        return name.startswith("linux/arm64/")
    return bool(name.startswith("org/lwjgl/") or VERSIONED.match(name))


def expected(source: bytes, jars: list[tuple[bytes, bool]]) -> list:
    """(5): the entries of S not in R, then A, a name already written skipped; each as
    (its source ZipInfo, its data)."""
    out, written = [], set()

    def take(z: ZipFile, keep) -> None:
        for info in z.infolist():
            if keep(info.filename) and info.filename not in written:
                written.add(info.filename)
                out.append((info, z.read(info)))

    take(ZipFile(io.BytesIO(source)), lambda name: not removed(name))
    for data, natives in jars:
        take(ZipFile(io.BytesIO(data)), lambda name, n=natives: added(name, n))
    return out


def entry_count(source: bytes, jars: list[tuple[bytes, bool]]) -> int:
    """The total of the `patch` bar: entries of S and of all the jars."""
    zips = [source] + [data for data, _ in jars]
    return sum(len(ZipFile(io.BytesIO(z)).infolist()) for z in zips)


def read_entries(data_or_path) -> list:
    """(ZipInfo, data) of every entry of a zip, in order."""
    with zipfile.ZipFile(data_or_path) as z:
        return [(info, z.read(info)) for info in z.infolist()]
