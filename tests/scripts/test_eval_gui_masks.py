"""scripts/eval_gui_masks.sh against specs/gui-masks.md and specs/gui-stack.md (Interface:
Script; gui-masks, then gui-stack) and the script preamble of specs/spark-env.md."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
EVALUATE = (*RUN, "mcpref.recognition.evaluate", "configs/gui-masks.yaml")
STACK = (*RUN, "mcpref.stacking.run", "configs/gui-stack.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="eval_gui_masks.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./eval_gui_masks.sh"])]
)
def test_evaluates_from_repository_root(spark, where, argv):
    """gui-masks, then gui-stack, both from the repository root."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [EVALUATE, STACK]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_evaluation_stops_the_script(spark):
    """gui-masks fails: the script fails without running gui-stack."""
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [EVALUATE]


def test_failing_stack_fails_the_script(spark):
    """gui-stack fails (set -e of the preamble): the script fails."""
    uv = spark.bin / "uv"
    uv.write_text(uv.read_text() + '[ "${5-}" != mcpref.stacking.run ] || exit 1\n')
    result = spark.run()
    assert result.returncode != 0 and result.events == [EVALUATE, STACK]
