"""`removed` ($R$) and `added` ($A$ without the repeat rule) of specs/engine.md (Notation): the
97 root names of the jar of [E], exactly its 64 LWJGL natives removed ([E], [R]); hand
cases for the prefixes, `re.match` at the start, `re.fullmatch` and the `/` rule."""

import pytest

from mcpref.engine.patch import added, removed

# [E]: the 64 root entries of LWJGL 3.2.2 natives for Linux x86-64, Windows and macOS, with
# their `.git` / `.sha1` files, as listed by the jar's central directory.
LWJGL_ROOTS = (
    "jemalloc.dll", "jemalloc.dll.git", "jemalloc.dll.sha1", "jemalloc32.dll",
    "jemalloc32.dll.git", "jemalloc32.dll.sha1", "libjemalloc.so", "libjemalloc.so.git",
    "libjemalloc.so.sha1", "libjemalloc.dylib.sha1",
    "OpenAL.dll", "OpenAL.dll.git", "OpenAL.dll.sha1", "OpenAL32.dll", "OpenAL32.dll.git",
    "OpenAL32.dll.sha1", "libopenal.so", "libopenal.so.git", "libopenal.so.sha1",
    "libopenal.dylib.sha1",
    "lwjgl_opengl.dll", "lwjgl_opengl.dll.sha1", "lwjgl_opengl32.dll",
    "lwjgl_opengl32.dll.sha1", "liblwjgl_opengl.so", "liblwjgl_opengl.so.sha1",
    "liblwjgl_opengl.dylib.sha1",
    "glfw.dll", "glfw.dll.git", "glfw.dll.sha1", "glfw32.dll", "glfw32.dll.git",
    "glfw32.dll.sha1", "libglfw.so", "libglfw.so.git", "libglfw.so.sha1",
    "libglfw_wayland.so", "libglfw_wayland.so.git", "libglfw_wayland.so.sha1",
    "libglfw.dylib.sha1",
    "lwjgl_stb.dll", "lwjgl_stb.dll.sha1", "lwjgl_stb32.dll", "lwjgl_stb32.dll.sha1",
    "liblwjgl_stb.so", "liblwjgl_stb.so.sha1", "liblwjgl_stb.dylib.sha1",
    "lwjgl_tinyfd.dll", "lwjgl_tinyfd.dll.sha1", "lwjgl_tinyfd32.dll",
    "lwjgl_tinyfd32.dll.sha1", "liblwjgl_tinyfd.so", "liblwjgl_tinyfd.so.sha1",
    "liblwjgl_tinyfd.dylib.sha1",
    "lwjgl.dll", "lwjgl.dll.git", "lwjgl.dll.sha1", "lwjgl32.dll", "lwjgl32.dll.git",
    "lwjgl32.dll.sha1", "liblwjgl.so", "liblwjgl.so.git", "liblwjgl.so.sha1",
    "liblwjgl.dylib.sha1",
)  # fmt: skip
# [E], [R]: the 33 other root entries, kept.
OTHER_ROOTS = (
    "Start.class", "MissionHandlers.xsd", "version.properties", "index.txt",
    "MissionInit.xsd", "Mission.xsd", "Types.xsd", "pack.mcmeta", "MissionEnded.xsd",
    "schemas.index", "pack.png", "log4j2.xml", "version.json",
    "yggdrasil_session_pubkey.der", "mcp_snapshot-20201028-1.16.3.zip", "libfliteWrapper.so",
    ".netbeans_automatic_build", "libjcocoa.dylib", "LICENSE", "Log4j-config.xsd",
    "Log4j-events.dtd", "Log4j-events.xsd", "Log4j-levels.xsd", "Log4j-charsets.properties",
    "module-info.class", "libjinput-linux.so", "libjinput-linux64.so", "jinput-dx8.dll",
    "jinput-dx8_64.dll", "jinput-raw.dll", "jinput-raw_64.dll", "jinput-wintab.dll",
    "libjinput-osx.jnilib",
)  # fmt: skip


def test_root_names_of_the_engine_jar():
    """97 distinct root names; exactly the 64 LWJGL natives are in R."""
    assert (len(LWJGL_ROOTS), len(OTHER_ROOTS)) == (64, 33)
    assert len(set(LWJGL_ROOTS + OTHER_ROOTS)) == 97
    assert [n for n in LWJGL_ROOTS + OTHER_ROOTS if removed(n)] == list(LWJGL_ROOTS)


def test_root_names_are_not_added():
    """No root name starts with `org/lwjgl/`, `META-INF/versions/` or `linux/arm64/`."""
    roots = LWJGL_ROOTS + OTHER_ROOTS
    assert not any(added(n, natives) for n in roots for natives in (False, True))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("org/lwjgl/", True),  # the prefix itself
        ("org/lwjgl/Version.class", True),
        ("org/lwjgl/system/linux/liblwjgl.so", True),
        ("org/lwjgl", False),  # no `/` after lwjgl
        ("org/lwjglx/Foo.class", False),
        ("net/org/lwjgl/Foo.class", False),  # a prefix, not a substring
        ("META-INF/versions/9/org/lwjgl/", True),
        ("META-INF/versions/10/org/lwjgl/system/MathUtil.class", True),
        ("META-INF/versions/123/org/lwjgl/X.class", True),
        ("META-INF/versions//org/lwjgl/X.class", False),  # [0-9]+: at least one digit
        ("META-INF/versions/9a/org/lwjgl/X.class", False),
        ("META-INF/versions/9/org/", False),
        ("META-INF/versions/9/module-info.class", False),
        ("x/META-INF/versions/9/org/lwjgl/X.class", False),  # re.match, not re.search
        ("META-INF/versions/9/org/lwjglx/X.class", False),
        ("liblwjgl_opengl32.so", True),  # (lib)? name (32)? .ext
        ("libOpenAL.so", True),
        ("openal32.dylib.git", True),
        ("liblwjgl.so.bak", False),  # re.fullmatch: nothing after the suffix
        ("xliblwjgl.so", False),  # nor before the name
        ("liblwjgl.so.sha1.git", False),
        ("liblwjgl.SO", False),
        ("LWJGL.dll", False),
        ("lwjgl_vulkan.dll", False),
        ("libglfw_wayland32.dll.sha1", True),
        ("lwjgl.jar", False),
        ("linux/x64/org/lwjgl/liblwjgl.so", False),  # natives with `/`: not root names
        ("natives/liblwjgl.so", False),
        ("windows/lwjgl.dll", False),
        ("linux/arm64/org/lwjgl/liblwjgl.so", False),
    ],
)
def test_removed_hand_cases(name, expected):
    assert removed(name) is expected


@pytest.mark.parametrize(
    "name, natives, expected",
    [
        ("org/lwjgl/", False, True),
        ("org/lwjgl/opengl/GL11.class", False, True),
        ("META-INF/versions/9/org/lwjgl/", False, True),
        ("META-INF/versions/10/org/lwjgl/system/MathUtil.class", False, True),
        ("x/META-INF/versions/9/org/lwjgl/X.class", False, False),  # re.match
        ("META-INF/versions/9/module-info.class", False, False),
        ("META-INF/MANIFEST.MF", False, False),
        ("META-INF/INDEX.LIST", False, False),
        ("org/", False, False),
        ("linux/arm64/org/lwjgl/liblwjgl.so", False, False),  # classes jars: no natives
        ("linux/arm64/", True, True),
        ("linux/arm64/org/lwjgl/liblwjgl.so", True, True),
        ("linux/arm64/org/lwjgl/opengl/liblwjgl_opengl.so.sha1", True, True),
        ("linux/", True, False),
        ("linux/arm64", True, False),
        ("linux/arm32/org/lwjgl/liblwjgl.so", True, False),
        ("x/linux/arm64/liblwjgl.so", True, False),
        ("org/lwjgl/Version.class", True, False),  # natives jars: only linux/arm64/
        ("META-INF/versions/9/org/lwjgl/X.class", True, False),
        ("META-INF/MANIFEST.MF", True, False),
    ],
)
def test_added_hand_cases(name, natives, expected):
    assert added(name, natives) is expected
