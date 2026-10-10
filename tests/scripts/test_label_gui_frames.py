"""scripts/label_gui_frames.sh against specs/gui-labels.md (Interface: Script) and the script
preamble of specs/spark-env.md."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
LABELS = (
    f"uv@{CFG['uv_version']}",
    "run",
    "--locked",
    "python",
    "-m",
    "mcpref.gui.labels",
)
LABELS += ("configs/gui-labels.yaml",)


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="label_gui_frames.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./label_gui_frames.sh"])]
)
def test_labels_from_repository_root(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [LABELS]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_labelling_fails_the_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [LABELS]
