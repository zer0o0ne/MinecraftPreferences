"""scripts/serve_vlm.sh against specs/vlm-client.md (Interface: Script) and the script preamble
of specs/spark-env.md."""

import os

import pytest

DROP_CACHES = ("sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches")


@pytest.fixture
def spark(make_script):
    return make_script("serve_vlm.sh")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./serve_vlm.sh"])]
)
def test_frees_page_cache_then_serves_from_repository_root(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [DROP_CACHES, spark.vlm_server("serve")]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


@pytest.mark.parametrize("command, calls", [("sudo", 1), ("run", 2)])
def test_failing_command_fails_the_script(spark, command, calls):
    spark.fail = command
    result = spark.run()
    assert result.returncode != 0
    assert result.events == [DROP_CACHES, spark.vlm_server("serve")][:calls]
