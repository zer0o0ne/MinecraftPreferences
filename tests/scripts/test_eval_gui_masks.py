"""scripts/eval_gui_masks.sh against specs/gui-masks.md (Interface: Script) and the script
preamble of specs/spark-env.md."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
EVALUATE = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
EVALUATE += ("mcpref.gui.evaluate", "configs/gui-masks.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="eval_gui_masks.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./eval_gui_masks.sh"])]
)
def test_evaluates_from_repository_root(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [EVALUATE]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_evaluation_fails_the_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [EVALUATE]
