"""scripts/pretrain.sh against specs/train-index.md (Interface: Script), specs/policy.md
(Interface: the line of `main`, (13)) and the script preamble of specs/spark-env.md, with a
fake uv."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
LOOP = (*RUN, "mcpref.training.pretrain.run", "configs/training/pretrain/config.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="pretrain.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./pretrain.sh"])]
)
def test_selected_pipeline_from_repository_root(spark, where, argv):
    """The common entry point owns index construction and mode dispatch."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [LOOP]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_pipeline_fails_the_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [LOOP]
