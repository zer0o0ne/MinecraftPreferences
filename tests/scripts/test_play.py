"""The play wrapper uses runtime defaults or one alternative config path."""

import os

import pytest


@pytest.mark.parametrize("args", [[], ["my runtime.yaml"]])
def test_play_wrapper(make_script, args):
    spark = make_script("play.sh")
    result = spark.run(argv=[str(spark.script), *args])
    assert result.returncode == 0, result.stderr
    assert result.events == [
        (
            "uv@0.13.2",
            "run",
            "--locked",
            "python",
            "-m",
            "mcpref.runtime.play.run",
            *args,
        )
    ]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_play_failure_propagates(make_script):
    spark = make_script("play.sh")
    spark.fail = "mcpref.runtime.play.run"
    assert spark.run().returncode != 0
