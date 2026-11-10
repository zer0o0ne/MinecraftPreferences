"""scripts/check_env.sh: the standard prelude and exact locked smoke command of specs/env.md."""

import os

import pytest

RUN = (
    "uv@0.13.2", "run", "--locked", "python", "-m",
    "mcpref.runtime.env.check", "configs/runtime/env.yaml",
)  # fmt: skip


@pytest.fixture
def spark(make_script):
    return make_script("check_env.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./check_env.sh"])]
)
def test_runs_smoke_from_repository_root(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [RUN]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_smoke_failure_fails_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [RUN]
