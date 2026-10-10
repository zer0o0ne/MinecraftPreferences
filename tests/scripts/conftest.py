"""Fake DGX Spark for the scripts: Tests items of specs/spark-env.md and specs/vlm-client.md."""

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIRS = ("repo/scripts", "repo/configs", "home/.local/bin", "sysbin", "ctl", "cwd")
STEPS = ["config", "arch", "driver", "docker", "uv", "python", "sync", "vlm"]
NVIDIA_SMI_ARGS = ("--query-gpu=driver_version", "--format=csv,noheader")
WITH_NVIDIA = (
    '{"io.containerd.runc.v2":{"path":"runc"},'
    '"nvidia":{"args":[],"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
)
RECORD = r"""{ printf 'CALL\037%s\037%s' @NAME@ "$PWD"; for a in "$@"; do printf '\037%s' "$a"; done; echo; } >> @LOG@
"""
DOCKER = """[ "$(cat @CTL@/docker)" = ok ] || { echo 'permission denied while trying to connect to the Docker daemon socket' >&2; exit 1; }
cat @CTL@/runtimes
"""
CURL = r"""[ "$(cat @CTL@/fail)" != curl ] || exit 22
[[ ${!#} =~ ^https://astral\.sh/uv/([^/]+)/install\.sh$ ]] || exit 22
echo "V=${BASH_REMATCH[1]}"
cat @CTL@/install.sh
"""
UV = r"""[ "$1" != --version ] || echo "uv $V (fake 2026-10-10)"
[ "$1" != "$(cat @CTL@/fail)" ] || exit 1
"""
INSTALL_SH = r"""{ printf 'CALL\037install.sh\037%s' "$PWD"; printf '\037UV_INSTALL_DIR=%s' "${UV_INSTALL_DIR-<unset>}"; printf '\037UV_NO_MODIFY_PATH=%s\n' "${UV_NO_MODIFY_PATH-<unset>}"; } >> @LOG@
dir=${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}
mkdir -p "$dir"
sed "s/^V=.*/V=$V/" @CTL@/uv > "$dir/uv"
chmod +x "$dir/uv"
[ "${UV_NO_MODIFY_PATH-}" = 1 ] || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
"""


FAKES = [  # (path under tmp, recorded name, body)
    ("sysbin/uname", "uname", "cat @CTL@/arch\n"),
    ("sysbin/nvidia-smi", "nvidia-smi", "cat @CTL@/driver\n"),
    ("sysbin/docker", "docker", DOCKER),
    ("sysbin/curl", "PATH-curl", "exit 1\n"),
    ("sysbin/sudo", "sudo", '[ "$(cat @CTL@/fail)" != sudo ] || exit 1\n'),
    ("home/.local/bin/curl", "curl", CURL),
]


@dataclass
class Result:
    returncode: int
    stderr: str
    events: list
    cwds: set


def parse(log: str) -> tuple[list, set]:
    """Stdout headers `== <name>` as ("==", name), fake calls as (tool, *args), in order."""
    events, cwds = [], set()
    for line in log.splitlines():
        if line.startswith("== "):
            events.append(("==", line[3:]))
        elif line.startswith("CALL\x1f"):
            _, tool, cwd, *args = line.split("\x1f")
            events.append((tool, *args))
            cwds.add(cwd)
    return events, cwds


class Spark:
    """Temp repository with a copy of `script`, a fake HOME and recording fakes.

    uname, nvidia-smi, docker and sudo sit on the caller's PATH (steps 1-3 run before
    step 4 prepends $HOME/.local/bin); curl and uv only in $HOME/.local/bin.
    curl on the caller's PATH is a guard that records and fails.
    """

    def __init__(self, tmp: Path, cfg: dict, script: str = "setup_env.sh"):
        self.tmp, self.cfg, self.repo, self.home = tmp, cfg, tmp / "repo", tmp / "home"
        self.script = self.repo / "scripts" / script
        self.bin, self.ctl, self.log = (
            self.home / ".local/bin",
            tmp / "ctl",
            tmp / "log",
        )
        for d in DIRS:
            (tmp / d).mkdir(parents=True)
        shutil.copy2(REPO / "scripts" / script, self.script)
        self.write_config(cfg)
        self.write_fakes()
        self.set_uv(cfg["uv_version"])
        self.arch, self.driver, self.docker = "aarch64", cfg["driver_min"], "ok"
        self.runtimes, self.fail = WITH_NVIDIA, ""
        self.env = {
            "HOME": str(self.home),
            "PATH": f"{tmp / 'sysbin'}:/usr/bin:/bin",
            "USER": "fakeuser",
            "XDG_BIN_HOME": str(tmp / "xdg-bin"),
            "LANG": "C.UTF-8",
        }

    def subst(self, text: str) -> str:
        log, ctl = shlex.quote(str(self.log)), shlex.quote(str(self.ctl))
        return text.replace("@LOG@", log).replace("@CTL@", ctl)

    def write(self, rel: str, body: str) -> None:
        path = self.tmp / rel
        path.write_text("#!/bin/bash\n" + self.subst(body))
        path.chmod(0o755)

    def write_fakes(self) -> None:
        for rel, name, body in FAKES:
            self.write(rel, RECORD.replace("@NAME@", name) + body)
        self.write("ctl/uv", "V=\n" + RECORD.replace("@NAME@", '"uv@$V"') + UV)
        (self.ctl / "install.sh").write_text(self.subst(INSTALL_SH))

    def write_config(self, cfg: dict) -> None:
        text = "".join(f"{k}: {v}\n" for k, v in cfg.items())
        (self.repo / "configs/spark-env.yaml").write_text(text)

    def set_uv(self, version: str | None) -> None:
        """Fake uv of `version` in $HOME/.local/bin; None removes it."""
        (self.bin / "uv").unlink(missing_ok=True)
        if version is not None:
            text = re.sub(r"(?m)^V=.*$", f"V={version}", (self.ctl / "uv").read_text())
            (self.bin / "uv").write_text(text)
            (self.bin / "uv").chmod(0o755)

    def run(self, cwd: Path | None = None, argv: list | None = None) -> Result:
        for key in ("arch", "driver", "docker", "runtimes", "fail"):
            (self.ctl / key).write_text(f"{getattr(self, key)}\n")
        self.log.write_text("")
        with self.log.open("a") as out:
            proc = subprocess.run(
                argv or [str(self.script)],
                cwd=cwd or self.tmp / "cwd",
                env=self.env,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
        return Result(proc.returncode, proc.stderr, *parse(self.log.read_text()))

    def home_files(self) -> dict:
        return {p: p.read_bytes() for p in self.home.rglob("*") if p.is_file()}

    def uv_check(self, version: str) -> list:
        return [(f"uv@{version}", "--version")]

    def uv_install(self) -> list:
        """(4): pinned install.sh piped to sh with UV_INSTALL_DIR and UV_NO_MODIFY_PATH=1."""
        url = f"https://astral.sh/uv/{self.cfg['uv_version']}/install.sh"
        env = (f"UV_INSTALL_DIR={self.home}/.local/bin", "UV_NO_MODIFY_PATH=1")
        return [("curl", "-LsSf", url), ("install.sh", *env)]

    def vlm_server(self, command: str) -> tuple:
        """`uv run --locked python -m mcpref.vlm.server <command> configs/vlm.yaml`."""
        uv = f"uv@{self.cfg['uv_version']}"
        return (
            uv,
            "run",
            "--locked",
            "python",
            "-m",
            "mcpref.vlm.server",
            command,
            "configs/vlm.yaml",
        )

    def expected(self, uv_calls: list, last: str = "vlm") -> list:
        """Steps (0)..`last` of Behaviour: header `== <name>`, then the step's calls."""
        v, py = self.cfg["uv_version"], self.cfg["python_version"]
        calls = {
            "config": [],
            "arch": [("uname", "-m")],
            "driver": [("nvidia-smi", *NVIDIA_SMI_ARGS)],
            "docker": [("docker", "info", "--format", "{{json .Runtimes}}")],
            "uv": uv_calls,
            "python": [(f"uv@{v}", "python", "install", py)],
            "sync": [(f"uv@{v}", "sync", "--locked", "--python", py)],
            "vlm": [self.vlm_server("prepare")],
        }
        return [
            e for s in STEPS[: STEPS.index(last) + 1] for e in [("==", s), *calls[s]]
        ]


@pytest.fixture
def make_spark(tmp_path):
    return lambda cfg, script="setup_env.sh": Spark(tmp_path, cfg, script)


@pytest.fixture
def make_script(make_spark):
    """Spark for a script run after setup_env.sh: uv installed in $HOME/.local/bin."""
    cfg = {
        "driver_min": "575.64.05",
        "python_version": "3.10.18",
        "uv_version": "0.13.2",
    }
    return lambda script: make_spark(cfg, script)
