"""scripts/download_data.sh: the step of specs/gui-screens.md (Interface)."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
TEXTURES = ("uv@0.13.2", "run", "--locked", "python", "-m", "mcpref.gui.textures")
TEXTURES += ("configs/gui-screens.yaml",)


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./download_data.sh"])]
)
def test_first_step_downloads_textures_from_repository_root(make_spark, where, argv):
    spark = make_spark(CFG, "download_data.sh")
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.events[:1] == [TEXTURES], result.stderr
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}
