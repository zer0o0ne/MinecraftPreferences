"""scripts/pretrain.sh against specs/train-index.md (Interface: Script) and the script
preamble of specs/spark-env.md, with a fake uv."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
INDEX = (*RUN, "mcpref.pretrain.index", "configs/train-index.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="pretrain.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./pretrain.sh"])]
)
def test_index_from_repository_root(spark, where, argv):
    """train-index, from the repository root, and nothing else."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [INDEX]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_index_fails_the_script(spark):
    """train-index fails: the script fails."""
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [INDEX]
