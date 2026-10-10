"""scripts/stop_vlm.sh against specs/vlm-client.md (Interface: Script) and the script preamble
of specs/spark-env.md."""

import os

import pytest


@pytest.fixture
def spark(make_script):
    return make_script("stop_vlm.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./stop_vlm.sh"])]
)
def test_stops_from_repository_root(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [spark.vlm_server("stop")]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_stop_fails_the_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [spark.vlm_server("stop")]
