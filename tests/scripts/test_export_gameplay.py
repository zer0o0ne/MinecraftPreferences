"""The video wrapper needs no settings arguments; only a config path is optional."""

import os

import pytest


@pytest.mark.parametrize("args", [[], ["my config.yaml"]])
def test_export_wrapper(make_script, args):
    spark = make_script("export_gameplay.sh")
    result = spark.run(argv=[str(spark.script), *args])
    assert result.returncode == 0, result.stderr
    assert result.events == [
        (
            "uv@0.13.2",
            "run",
            "--locked",
            "python",
            "-m",
            "mcpref.runtime.video.export",
            *args,
        )
    ]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_export_failure_propagates(make_script):
    spark = make_script("export_gameplay.sh")
    spark.fail = "mcpref.runtime.video.export"
    assert spark.run().returncode != 0
