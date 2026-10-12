"""Fake DGX Spark for the scripts: Tests items of specs/spark-env.md, specs/vlm-client.md
and specs/vpt-gui-frames.md (setup_env.sh step (7))."""

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DIRS = ("repo/scripts", "repo/configs", "home/.local/bin", "sysbin", "ctl", "cwd")
STEPS = ["config", "arch", "driver", "docker", "uv", "python", "sync", "ffmpeg", "vlm"]
NVIDIA_SMI_ARGS = ("--query-gpu=driver_version", "--format=csv,noheader")
# Step (7) of specs/vpt-gui-frames.md: the nine packages, in the order of its dpkg-query
# call; `apt-get install` and `apt-mark hold` may take them in any order (see `canonical`).
DEBS = ("ffmpeg", "libavcodec60", "libavdevice60", "libavfilter9", "libavformat60",
        "libavutil58", "libpostproc57", "libswresample4", "libswscale7")  # fmt: skip
DPKG_QUERY = ("dpkg-query", "-W", r"-f=${db:Status-Status} ${Version}\n", *DEBS)
UNORDERED = (("apt-get", "install"), ("apt-mark", "hold"))
DOCKER_INFO = "Server Version: fake-docker-info"
RECORD = r"""{ printf 'CALL\037%s\037%s' @NAME@ "$PWD"; for a in "$@"; do printf '\037%s' "$a"; done; echo; } >> @LOG@
"""
DOCKER = r"""[ "$*" = info ] || { echo "fake docker: unsupported $*" >&2; exit 125; }
[ "$(cat @CTL@/docker)" = ok ] || { echo 'permission denied while trying to connect to the Docker daemon socket' >&2; exit 1; }
echo '@INFO@'
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
# sudo runs the fake apt tools as root; any other command (serve_vlm.sh) is only recorded.
SUDO = r"""[ "$(cat @CTL@/fail)" != sudo ] || exit 1
case $1 in apt-get | apt-mark) export FAKE_ROOT=1; exec "$@" ;; esac
"""
AS_ROOT = r"""[ -n "${FAKE_ROOT-}" ] || { echo "E: @NAME@: Permission denied, are you root?" >&2; exit 100; }
[ "$(cat @CTL@/fail)" != "@NAME@ $1" ] || exit 100
"""
APT_GET = r"""[ "$1" = install ] || exit 0
for a in "${@:2}"; do
  [[ $a == -* ]] && continue
  awk -v p="${a%%=*}" '$1 != p' @CTL@/dpkg > @CTL@/dpkg.new
  echo "${a%%=*} installed ${a#*=}" >> @CTL@/dpkg.new
  mv @CTL@/dpkg.new @CTL@/dpkg
done
"""
# dpkg-query -W -f=FORMAT PACKAGE...: one line per database entry (`name` or `name:arch`)
# matching a requested name, in name order; an unmatched name is reported and exits 1.
DPKG = r"""[ "$1 $2" = '-W -f=${db:Status-Status} ${Version}\n' ] || { echo "fake dpkg-query: unsupported $*" >&2; exit 2; }
shift 2
sort @CTL@/dpkg | awk -v pkgs="$*" '
  BEGIN { n = split(pkgs, want, " ") }
  { split($1, name, ":"); for (i = 1; i <= n; i++) if (name[1] == want[i]) { print $2 " " $3; found[i] = 1; break } }
  END { for (i = 1; i <= n; i++) if (!found[i]) { print "dpkg-query: no packages found matching " want[i] > "/dev/stderr"; rc = 1 }; exit rc }'
"""


FAKES = [  # (path under tmp, recorded name, body)
    ("sysbin/uname", "uname", "cat @CTL@/arch\n"),
    ("sysbin/nvidia-smi", "nvidia-smi", "cat @CTL@/driver\n"),
    ("sysbin/docker", "docker", DOCKER),
    ("sysbin/curl", "PATH-curl", "exit 1\n"),
    ("sysbin/sudo", "sudo", SUDO),
    ("sysbin/apt-get", "apt-get", AS_ROOT + APT_GET),
    ("sysbin/apt-mark", "apt-mark", AS_ROOT),
    ("sysbin/dpkg-query", "dpkg-query", DPKG),
    ("home/.local/bin/curl", "curl", CURL),
]


@dataclass
class Result:
    returncode: int
    stderr: str
    events: list
    cwds: set
    log: str


def canonical(event: tuple) -> tuple:
    """Step (7) fixes the package set of `apt-get install` and `apt-mark hold`, not its
    order: their non-option arguments sorted (also behind sudo); options keep their order."""
    i = int(event[0] == "sudo")
    if event[i : i + 2] not in UNORDERED:
        return event
    args = event[i + 2 :]
    options = [a for a in args if a.startswith("-")]
    return (*event[: i + 2], *options, *sorted(a for a in args if a not in options))


def parse(log: str) -> tuple[list, set]:
    """Stdout headers `== <name>` as ("==", name), fake calls as canonical (tool, *args),
    in order."""
    events, cwds = [], set()
    for line in log.splitlines():
        if line.startswith("== "):
            events.append(("==", line[3:]))
        elif line.startswith("CALL\x1f"):
            _, tool, cwd, *args = line.split("\x1f")
            events.append(canonical((tool, *args)))
            cwds.add(cwd)
    return events, cwds


class Spark:
    """Temp repository with a copy of `script`, a fake HOME and recording fakes.

    uname, nvidia-smi, docker, sudo and the dpkg/apt tools sit on the caller's PATH (steps
    1-3 run before step 4 prepends $HOME/.local/bin); curl and uv only in $HOME/.local/bin.
    curl on the caller's PATH is a guard that records and fails. The dpkg database starts
    with the nine packages of step (7) installed at `ffmpeg_deb_version`, if `cfg` has it.
    """

    docker_info = DOCKER_INFO  # what the fake `docker info` prints on stdout

    def __init__(self, tmp: Path, cfg: dict, script: str = "setup_env.sh"):
        self.tmp, self.cfg, self.repo, self.home = tmp, cfg, tmp / "repo", tmp / "home"
        self.script = self.repo / "scripts" / script
        self.bin = self.home / ".local/bin"
        self.ctl, self.log = tmp / "ctl", tmp / "log"
        for d in DIRS:
            (tmp / d).mkdir(parents=True)
        shutil.copy2(REPO / "scripts" / script, self.script)
        self.write_config(cfg)
        self.write_fakes()
        self.set_uv(cfg["uv_version"])
        self.set_debs(self.pinned() if "ffmpeg_deb_version" in cfg else {})
        self.arch, self.driver, self.docker = "aarch64", cfg["driver_min"], "ok"
        self.fail = ""
        self.env = {
            "HOME": str(self.home),
            "PATH": f"{tmp / 'sysbin'}:/usr/bin:/bin",
            "USER": "fakeuser",
            "XDG_BIN_HOME": str(tmp / "xdg-bin"),
            "LANG": "C.UTF-8",
        }

    def subst(self, text: str) -> str:
        log, ctl = shlex.quote(str(self.log)), shlex.quote(str(self.ctl))
        text = text.replace("@INFO@", DOCKER_INFO)
        return text.replace("@LOG@", log).replace("@CTL@", ctl)

    def write(self, rel: str, body: str) -> None:
        path = self.tmp / rel
        path.write_text("#!/bin/bash\n" + self.subst(body))
        path.chmod(0o755)

    def write_fakes(self) -> None:
        for rel, name, body in FAKES:
            self.write(rel, (RECORD + body).replace("@NAME@", name))
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

    def set_debs(self, debs: dict) -> None:
        """The fakes' dpkg database: package (`name` or `name:arch`) -> (status, version)."""
        text = "".join(f"{p} {status} {v}\n" for p, (status, v) in debs.items())
        (self.ctl / "dpkg").write_text(text)

    def pinned(self, **changes) -> dict:
        """The nine packages of step (7) installed at `ffmpeg_deb_version`, with `changes`
        (package -> (status, version), or None for a package dpkg does not know)."""
        version = self.cfg["ffmpeg_deb_version"]
        debs = {p: ("installed", version) for p in DEBS} | changes
        return {p: entry for p, entry in debs.items() if entry is not None}

    def run(self, cwd: Path | None = None, argv: list | None = None) -> Result:
        for key in ("arch", "driver", "docker", "fail"):
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
        log = self.log.read_text()
        return Result(proc.returncode, proc.stderr, *parse(log), log)

    def home_files(self) -> dict:
        return {p: p.read_bytes() for p in self.home.rglob("*") if p.is_file()}

    def uv_check(self, version: str) -> list:
        return [(f"uv@{version}", "--version")]

    def uv_install(self) -> list:
        """(4): pinned install.sh piped to sh with UV_INSTALL_DIR and UV_NO_MODIFY_PATH=1."""
        url = f"https://astral.sh/uv/{self.cfg['uv_version']}/install.sh"
        env = (f"UV_INSTALL_DIR={self.home}/.local/bin", "UV_NO_MODIFY_PATH=1")
        return [("curl", "-LsSf", url), ("install.sh", *env)]

    def ffmpeg_install(self) -> list:
        """Step (7) when the check fails: the dpkg-query check, then each sudo command
        followed by the fake it runs as root (canonical)."""
        pins = [f"{p}={self.cfg['ffmpeg_deb_version']}" for p in DEBS]
        root = [
            ("apt-get", "update"),
            ("apt-get", "install", "-y", "--allow-downgrades", *pins),
            ("apt-mark", "hold", *DEBS),
        ]
        calls = [e for c in root for e in (("sudo", *c), c)]
        return [DPKG_QUERY] + [canonical(e) for e in calls]

    def vlm_server(self, command: str) -> tuple:
        """`uv run --locked python -m mcpref.vlm.server <command> configs/vlm.yaml`."""
        uv = f"uv@{self.cfg['uv_version']}"
        run = ("run", "--locked", "python", "-m", "mcpref.vlm.server")
        return (uv, *run, command, "configs/vlm.yaml")

    def expected(
        self, uv_calls: list, last: str = "vlm", ffmpeg: list | None = None
    ) -> list:
        """Steps (0)..`last` of Behaviour: header `== <name>`, then the step's calls;
        step (7) is the dpkg-query check alone unless `ffmpeg` is given."""
        v, py = self.cfg["uv_version"], self.cfg["python_version"]
        calls = {
            "config": [],
            "arch": [("uname", "-m")],
            "driver": [("nvidia-smi", *NVIDIA_SMI_ARGS)],
            "docker": [("docker", "info")],
            "uv": uv_calls,
            "python": [(f"uv@{v}", "python", "install", py)],
            "sync": [(f"uv@{v}", "sync", "--locked", "--python", py)],
            "ffmpeg": [DPKG_QUERY] if ffmpeg is None else ffmpeg,
            "vlm": [self.vlm_server("prepare")],
        }
        steps = STEPS[: STEPS.index(last) + 1]
        return [e for s in steps for e in [("==", s), *calls[s]]]


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
