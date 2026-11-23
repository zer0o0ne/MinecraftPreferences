"""Real subprocess fakes without a newly generated executable for every test."""

import subprocess
import sys
from functools import cache
from pathlib import Path

LAUNCHER = Path(__file__).with_name("fake_program.sh")


@cache
def ready() -> None:
    """Finish first-launch script startup before testing decoder deadlines."""
    subprocess.run([str(LAUNCHER), "--ready"], check=True, timeout=10)


def program(path: Path, script: Path, control: Path, log: Path) -> None:
    """Bind a shared launcher to this test's interpreter, script and state files."""
    ready()
    arguments = [sys.executable, str(script), str(control), str(log)]
    path.with_name(path.name + ".args").write_text("\n".join(arguments) + "\n")
    path.symlink_to(LAUNCHER)
