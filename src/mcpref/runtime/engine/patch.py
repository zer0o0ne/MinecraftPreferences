"""(5) of specs/engine.md: the engine jar with LWJGL replaced by the jars of (2)."""

import io
import re
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

VERSIONED = re.compile(r"META-INF/versions/[0-9]+/org/lwjgl/")
NATIVE = re.compile(
    r"(lib)?(lwjgl(_opengl|_stb|_tinyfd)?|glfw(_wayland)?|openal|OpenAL|jemalloc)"
    r"(32)?\.(so|dll|dylib)(\.git|\.sha1)?"
)


def lwjgl_class(name: str) -> bool:
    """Notation R, A: under org/lwjgl/ or META-INF/versions/<n>/org/lwjgl/."""
    return name.startswith("org/lwjgl/") or VERSIONED.match(name) is not None


def removed(name: str) -> bool:
    """Notation R: an LWJGL class entry or a root LWJGL native with its .git/.sha1 (NATIVE
    holds no `/`, so its full match is a name without `/`)."""
    return lwjgl_class(name) or NATIVE.fullmatch(name) is not None


def added(name: str, natives: bool) -> bool:
    """Notation A without the repeat rule: linux/arm64/ entries of a natives jar, LWJGL
    class entries of a classes jar."""
    return name.startswith("linux/arm64/") if natives else lwjgl_class(name)


def patch(source: Path, jars: Sequence[tuple[bytes, bool]], out: Path) -> None:
    """(5) up to the zip written to `out`: S without R, then A in jar order, a name
    already written skipped; each entry by writestr with the source's own ZipInfo;
    `jars` = (bytes, is natives jar) in order."""
    with zipfile.ZipFile(source) as s, zipfile.ZipFile(out, "w") as new:
        opened = [
            (zipfile.ZipFile(io.BytesIO(data)), natives) for data, natives in jars
        ]
        total = len(s.infolist()) + sum(len(j.infolist()) for j, _ in opened)
        with tqdm(total=total, desc="patch", unit="entry", file=sys.stderr) as bar:
            written = set()
            for info in s.infolist():
                if not removed(info.filename):
                    new.writestr(info, s.read(info))
                    written.add(info.filename)
                bar.update()
            for jar, natives in opened:
                for info in jar.infolist():
                    if added(info.filename, natives) and info.filename not in written:
                        new.writestr(info, jar.read(info))
                        written.add(info.filename)
                    bar.update()
