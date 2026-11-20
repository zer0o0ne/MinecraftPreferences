"""Atomic progress-observation hook for the pinned engine; runtime-stagnation spec."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from tqdm import tqdm

from mcpref.infrastructure.cli.failure import failed
from mcpref.infrastructure.shards.store import publish
from mcpref.runtime.engine.config import engine_jar, load_config

ENV = "com/minerl/multiagent/env/EnvServer.class"
HELPER = "mcpref/runtime/ActionProgress.class"
ORIGINAL = "META-INF/mcpref/EnvServer.original.class"
MARKER = "META-INF/mcpref/action-progress.json"
SOURCES = Path(__file__).with_suffix("")
JAVA_SOURCES = tuple(
    SOURCES / name for name in ("ActionProgress.java", "PatchTelemetry.java")
)
# Audited mcprec-6.13 bytecode, unaffected by the existing LWJGL patch.
SOURCE_HASHES = {
    ENV: "dadfe9aaf364da6ec4237ee66a9a021af29e986a2e6870d9367d68d4f0db3cc5",
    "net/minecraft/client/multiplayer/PlayerController.class": "578016af7668c561a9c4cade765f8fd5cb4472abd4e07e9afcec623d838baa5e",
    "net/minecraft/entity/LivingEntity.class": "dc2655e1dcb6dee0e453b3e49000ba5b92d03282d9896839ebd9523c78956d77",
    "net/minecraft/item/ItemStack.class": "ab39b28bea93ddc7ceb981e46d684f01be51b4378d7a0d315254b44eb0164ab4",
    "net/minecraft/item/BowItem.class": "3136318b35a421d9743537e29ae585a18f380f33b0f77f997a3eba59418d25b5",
    "net/minecraft/item/CrossbowItem.class": "8132206b475f629368441dde2130d4b64f2bc06199c12d88a4f6cc89b28236dd",
    "net/minecraft/item/TridentItem.class": "1addfc32213c30ec1cd77a3ffc16cf9671b7e537b106e187f2ff5777f041aedc",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _implementation() -> str:
    return _sha(b"".join(path.read_bytes() for path in JAVA_SOURCES))


def _incompatible(detail: str) -> RuntimeError:
    return RuntimeError(f"incompatible engine telemetry: {detail}")


def _inspect(archive: zipfile.ZipFile) -> tuple[bytes, bool]:
    """Verify pinned inputs and every previously installed output before reuse."""
    names = archive.namelist()
    if any(names.count(name) > 1 for name in (ENV, HELPER, ORIGINAL, MARKER)):
        raise _incompatible("duplicate telemetry entry")
    marker = json.loads(archive.read(MARKER)) if MARKER in names else None
    original = archive.read(ORIGINAL if marker is not None else ENV)
    for name, expected in SOURCE_HASHES.items():
        data = original if name == ENV else archive.read(name)
        if _sha(data) != expected:
            raise _incompatible(f"unexpected source {name}")
    if marker is None:
        if HELPER in names or ORIGINAL in names:
            raise _incompatible("unmarked telemetry classes")
        return original, False
    if marker.get("source") != SOURCE_HASHES or set(marker.get("classes", {})) != {
        ENV,
        HELPER,
    }:
        raise _incompatible("invalid installation marker")
    for name, expected in marker["classes"].items():
        if _sha(archive.read(name)) != expected:
            raise _incompatible(f"modified installed class {name}")
    return original, marker.get("implementation") == _implementation()


def _compile(jar: Path, work: Path, original: bytes) -> dict[str, bytes]:
    """Compile against the engine's own Minecraft/Gson/ASM, then insert one hook."""
    source, patched = work / "EnvServer.class", work / "EnvServer.patched.class"
    source.write_bytes(original)
    _run(
        [
            "javac",
            "-proc:none",
            "-g:none",
            "-source",
            "8",
            "-target",
            "8",
            "-encoding",
            "UTF-8",
            "-cp",
            str(jar),
            "-d",
            str(work),
            *(str(path) for path in JAVA_SOURCES),
        ],
    )
    _run(
        [
            "java",
            "-cp",
            os.pathsep.join((str(work), str(jar))),
            "mcpref.runtime.PatchTelemetry",
            str(source),
            str(patched),
        ],
    )
    return {ENV: patched.read_bytes(), HELPER: (work / HELPER).read_bytes()}


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"engine telemetry {command[0]} failed: {error.stderr.strip()}"
        ) from error


def _write(
    archive: zipfile.ZipFile, out: Path, original: bytes, classes: dict[str, bytes]
) -> None:
    """Preserve all unrelated entries and retain the pristine class for upgrades."""
    marker = {
        "source": SOURCE_HASHES,
        "implementation": _implementation(),
        "classes": {name: _sha(data) for name, data in classes.items()},
    }
    added = {
        HELPER: classes[HELPER],
        ORIGINAL: original,
        MARKER: json.dumps(marker, sort_keys=True).encode(),
    }
    entries = archive.infolist()
    with (
        zipfile.ZipFile(out, "w") as output,
        tqdm(
            total=len(entries) + len(added),
            desc="engine telemetry",
            unit="entry",
            file=sys.stderr,
        ) as bar,
    ):
        for info in entries:
            if info.filename not in added:
                data = classes[ENV] if info.filename == ENV else archive.read(info)
                output.writestr(info, data)
            bar.update()
        for name, data in added.items():
            output.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
            bar.update()


def install(jar: Path) -> bool:
    """Install or upgrade atomically; verified current installations are unchanged."""
    jar = jar.resolve()
    with zipfile.ZipFile(jar) as archive:
        original, current = _inspect(archive)
        if current:
            return False
        with tempfile.TemporaryDirectory(prefix=".telemetry-", dir=jar.parent) as tmp:
            work = Path(tmp)
            classes = _compile(jar, work, original)
            out = work / "engine.jar"
            _write(archive, out, original, classes)
            with zipfile.ZipFile(out) as check:
                if not _inspect(check)[1]:
                    raise _incompatible("incomplete staged installation")
            os.chmod(out, jar.stat().st_mode)
            publish(out, jar)
    return True


def main(argv: list[str]) -> int:
    """Install telemetry using the existing engine configuration."""
    parser = argparse.ArgumentParser(prog="python -m mcpref.runtime.engine.telemetry")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        changed = install(engine_jar(load_config(args.config)))
        print(
            "engine telemetry: installed"
            if changed
            else "engine telemetry: already done",
            file=sys.stderr,
        )
    except Exception as error:  # noqa: BLE001 (engine CLI reports all failures)
        return failed("mcpref.runtime.engine.telemetry", error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
