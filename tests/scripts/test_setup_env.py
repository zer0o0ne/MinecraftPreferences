"""scripts/setup_env.sh and the project files against specs/spark-env.md; step (7) against
specs/vpt-gui-frames.md (Interface: `scripts/setup_env.sh` step (7))."""

import importlib
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest
import tomli

DEV = {"pytest==9.1.1", "ruff==0.16.9", "pytest-timeout==2.4.0", "pytest-xdist==3.8.0"}
ADDOPTS = "-m 'not gpu and not slow' -n auto --timeout=10 --durations=10"
# CFG differs from the starting values (Notation) so that hard-coded values fail; the deb
# version has an epoch, a tilde and a plus, as Debian versions may.
CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2",
       "ffmpeg_deb_version": "9:8.0.1-1~test2+b1"}  # fmt: skip
V, UV = CFG["ffmpeg_deb_version"], CFG["uv_version"]
# Step (7): database changes after which dpkg-query does not print nine lines `installed V`;
# libavdevice60, libavfilter9 and libpostproc57 are the three of the 2026-10-12 amendment.
NOT_PINNED = {
    "one package unknown": {"libswresample4": None},
    "one at another version": {"libavutil58": ("installed", "9:8.0.2-1")},
    "one at a longer version": {"libavformat60": ("installed", V + "build1")},
    "one removed, version kept": {"libswscale7": ("config-files", V)},
    "second architecture's copy": {"libavcodec60:armhf": ("installed", V)},
    "libavdevice60 unknown": {"libavdevice60": None},
    "libavfilter9 at another version": {"libavfilter9": ("installed", "9:8.0.2-1")},
    "libpostproc57 removed, version kept": {"libpostproc57": ("config-files", V)},
}


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG)


def assert_failed(result, events: list, step: str, *messages: str) -> None:
    """A failing step prints its own message, then the trap line; no later step runs."""
    trap = f"setup_env.sh: step {step} failed"
    lines = result.stderr.splitlines()
    assert result.returncode != 0
    assert result.events == events
    assert lines[-1:] == [trap] and lines.count(trap) == 1, result.stderr
    for message in messages:
        assert message in "\n".join(lines[:-1]), result.stderr


def read_config(path: Path) -> dict:
    """Notation: one `key: value` per line; value without spaces, quotes, comments."""
    lines = path.read_text().splitlines()
    assert all(re.fullmatch(r"[a-z_]+: [^\s\"'#]+", line) for line in lines), lines
    return dict(line.split(": ") for line in lines)


def test_steps_in_order_with_spec_arguments(spark):
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert result.events == spark.expected(spark.uv_check(UV))


def test_committed_config_drives_the_steps(make_spark, repo):
    spark = make_spark(read_config(repo / "configs/spark-env.yaml"))
    shutil.copy(repo / "configs/spark-env.yaml", spark.repo / "configs")
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert result.events == spark.expected(spark.uv_check(spark.cfg["uv_version"]))


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./setup_env.sh"])]
)
def test_runs_in_repository_root(spark, where, argv):
    spark.set_debs({})  # step (7) installs: sudo and the apt tools run too
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_absent_uv_is_installed(spark):
    spark.set_uv(None)
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert result.events == spark.expected(spark.uv_install())


@pytest.mark.parametrize("old", ["0.13.20", "0.13", "0.12.19"])  # uv_version = 0.13.2
def test_other_uv_version_is_replaced(spark, old):
    spark.set_uv(old)
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert result.events == spark.expected(spark.uv_check(old) + spark.uv_install())


def test_second_run_installs_nothing(spark):
    spark.set_uv(None)
    spark.set_debs({})
    first = spark.run()
    home = spark.home_files()
    second = spark.run()
    assert (first.returncode, second.returncode) == (0, 0), second.stderr
    install = spark.expected(spark.uv_install(), ffmpeg=spark.ffmpeg_install())
    assert first.events == install
    assert second.events == spark.expected(spark.uv_check(UV))
    assert spark.home_files() == home


def test_missing_config_file(spark):
    (spark.repo / "configs/spark-env.yaml").unlink()
    message = "setup_env.sh: missing configs/spark-env.yaml"
    assert_failed(spark.run(), spark.expected([], "config"), "config", message)


@pytest.mark.parametrize("key", list(CFG))
def test_missing_config_key(spark, key):
    spark.write_config({k: v for k, v in CFG.items() if k != key})
    message = f"setup_env.sh: missing {key} in configs/spark-env.yaml"
    assert_failed(spark.run(), spark.expected([], "config"), "config", message)


@pytest.mark.parametrize("arch", ["x86_64", "arm64", "aarch64_be"])
def test_arch_other_than_aarch64_fails(spark, arch):
    spark.arch = arch
    assert_failed(spark.run(), spark.expected([], "arch"), "arch")


# driver_min = 575.64.05; string order misjudges 575.100.01, 1000.0.0, 575.9.99, 99.99.99
@pytest.mark.parametrize("driver", ["575.64.05", "575.64.06", "575.100.01", "1000.0.0"])
def test_driver_at_least_driver_min_passes(spark, driver):
    spark.driver = driver
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert result.events == spark.expected(spark.uv_check(UV))


@pytest.mark.parametrize("driver", ["575.64.04", "575.9.99", "99.99.99"])
def test_driver_below_driver_min_fails(spark, driver):
    spark.driver = driver
    message = f"driver {driver} is older than driver_min {CFG['driver_min']}"
    assert_failed(spark.run(), spark.expected([], "driver"), "driver", message)


def test_docker_info_output_is_discarded(spark):
    result = spark.run()
    assert result.returncode == 0, result.stderr
    assert spark.docker_info not in result.log


def test_docker_inaccessible_asks_to_join_group(spark):
    spark.docker = "denied"
    messages = ("sudo usermod -aG docker $USER", "log in again")
    assert_failed(spark.run(), spark.expected([], "docker"), "docker", *messages)


def test_failed_download_fails_uv_step(spark):
    spark.set_uv(None)
    spark.fail = "curl"
    assert_failed(spark.run(), spark.expected(spark.uv_install()[:1], "uv"), "uv")


@pytest.mark.parametrize(
    "step, command", [("python", "python"), ("sync", "sync"), ("vlm", "run")]
)
def test_failed_uv_command_fails_its_step(spark, step, command):
    spark.fail = command
    events = spark.expected(spark.uv_check(UV), step)
    assert_failed(spark.run(), events, step)


@pytest.mark.parametrize("state", ["no package known", *NOT_PINNED])
def test_ffmpeg_not_pinned_is_installed_and_held_with_sudo(spark, state):
    spark.set_debs(spark.pinned(**NOT_PINNED[state]) if state in NOT_PINNED else {})
    result = spark.run()
    assert result.returncode == 0, result.stderr
    install = spark.ffmpeg_install()
    assert result.events == spark.expected(spark.uv_check(UV), ffmpeg=install)


@pytest.mark.parametrize(  # `calls`: the failing call is the last; sudo, then the apt tool
    "command, calls",
    [("sudo", 2), ("apt-get update", 3), ("apt-get install", 5), ("apt-mark hold", 7)],
)
def test_failed_ffmpeg_command_fails_its_step(spark, command, calls):
    spark.set_debs({})
    spark.fail = command
    ffmpeg = spark.ffmpeg_install()[:calls]
    events = spark.expected(spark.uv_check(UV), "ffmpeg", ffmpeg)
    assert_failed(spark.run(), events, "ffmpeg")


def test_committed_config_starting_values(repo):
    assert read_config(repo / "configs/spark-env.yaml") == {
        "uv_version": "0.12.19",
        "python_version": "3.10.20",
        "driver_min": "580.95.05",
        "ffmpeg_deb_version": "7:6.1.1-3ubuntu5",  # vpt-gui-frames: Library mapping
    }


@pytest.fixture
def pyproject(repo) -> dict:
    return tomli.loads((repo / "pyproject.toml").read_text())


def test_project_table(pyproject):
    project = pyproject["project"]
    assert (project["name"], project["version"]) == ("mcpref", "0.1.0")
    assert project["requires-python"] == "==3.10.*"
    assert {"numpy==1.26.4", "pyyaml==6.0.3"} <= set(project["dependencies"])
    pinned = r"[A-Za-z0-9._-]+(\[[A-Za-z0-9._,-]+\])?==[^\s,;*]+(\s*;.+)?"
    assert all(re.fullmatch(pinned, d) for d in project["dependencies"])


def test_build_system_and_package(pyproject, repo):
    assert pyproject["build-system"] == {
        "requires": ["uv_build==0.12.19"],
        "build-backend": "uv_build",
    }
    package = Path(importlib.import_module("mcpref").__file__).resolve()
    assert package == repo / "src/mcpref/__init__.py"


def test_dev_group(pyproject):
    assert {k: set(v) for k, v in pyproject["dependency-groups"].items()} == {
        "dev": DEV
    }


def test_uv_required_version_is_uv_version(pyproject):
    assert pyproject["tool"]["uv"]["required-version"] == "==0.12.19"


def test_pytest_tiers(pytestconfig):
    markers = {m.split(":")[0].strip() for m in pytestconfig.getini("markers")}
    assert {"gpu", "slow"} <= markers
    assert pytestconfig.getini("addopts") == shlex.split(ADDOPTS)
