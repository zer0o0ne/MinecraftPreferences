"""scripts/check_sim.sh against specs/engine.md (Interface: `scripts/check_sim.sh`) and the
script preamble of specs/spark-env.md, with a fake uv."""

import os

import pytest

CHECK = (
    "run",
    "--locked",
    "python",
    "-m",
    "mcpref.engine.check",
    "configs/engine.yaml",
)


@pytest.fixture
def spark(make_script):
    return make_script("check_sim.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./check_sim.sh"])]
)
def test_runs_check_from_repository_root(spark, where, argv):
    """`uv run --locked python -m mcpref.engine.check configs/engine.yaml` from the
    repository root, uv found in $HOME/.local/bin, and nothing else."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [(f"uv@{spark.cfg['uv_version']}", *CHECK)]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_check_fails_the_script(spark):
    spark.fail = "mcpref.engine.check"
    result = spark.run()
    assert result.returncode != 0
    assert result.events == [(f"uv@{spark.cfg['uv_version']}", *CHECK)]


def test_preamble(repo):
    """`set -euo pipefail`, the repository root, $HOME/.local/bin on PATH."""
    lines = (repo / "scripts/check_sim.sh").read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    preamble = [
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        'PATH="$HOME/.local/bin:$PATH"',
    ]
    assert [line for line in lines if not line.startswith("#")][:3] == preamble
