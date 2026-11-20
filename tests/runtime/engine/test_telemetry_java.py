"""Real pinned-engine compilation and bytecode verification on the Spark."""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from mcpref.runtime.engine.config import engine_jar, load_config
from mcpref.runtime.engine.telemetry import HELPER, _compile, _inspect


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_real_hook_targets_final_observation_only(tmp_path):
    jar = engine_jar(load_config(Path("configs/runtime/engine.yaml")))
    if not jar.is_file() or not shutil.which("javac") or not shutil.which("javap"):
        pytest.skip("needs the pinned engine and JDK on the Spark")
    with zipfile.ZipFile(jar) as archive:
        original, _ = _inspect(archive)
    classes = _compile(jar, tmp_path, original)
    assert classes[HELPER].startswith(b"\xca\xfe\xba\xbe")
    out = subprocess.run(
        ["javap", "-private", "-c", str(tmp_path / "EnvServer.patched.class")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    method = out.split("private java.lang.String getInfo(java.lang.String);", 1)[1]
    method = re.split(r"\n  (?:public|private|protected) ", method, maxsplit=1)[0]
    assert method.count("mcpref/runtime/ActionProgress.append") == 1
    assert method.count("JsonObject.toString") == 2
    instructions = [line.strip() for line in method.splitlines() if ":" in line]
    tail = "\n".join(instructions[-4:])
    assert re.search(r"\d+: dup\n.*invokestatic.*ActionProgress.append", tail)
    assert "JsonObject.toString" in instructions[-2]
    assert instructions[-1].endswith("areturn")
