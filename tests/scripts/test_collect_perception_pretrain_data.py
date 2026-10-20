"""scripts/collect_perception_pretrain_data.sh against specs/perception-data.md (Interface,
revision of 2026-10-20: textures, then `main` of clips/run.py) and the script preamble of
specs/spark-env.md, with a fake uv."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
TEXTURES = (*RUN, "mcpref.gui.textures", "configs/gui-screens.yaml")
COLLECT = (*RUN, "mcpref.clips.run", "configs/perception-data.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="collect_perception_pretrain_data.sh")


@pytest.mark.parametrize(
    "where, argv",
    [("cwd", None), ("repo/scripts", ["./collect_perception_pretrain_data.sh"])],
)
def test_textures_then_main_from_repository_root(spark, where, argv):
    """The textures, then the clips, both from the repository root."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [TEXTURES, COLLECT]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_textures_stop_the_script(spark):
    """The textures fail: the script fails without running main."""
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [TEXTURES]


def test_failing_main_fails_the_script(spark):
    """main fails (set -e of the preamble): the script fails."""
    uv = spark.bin / "uv"
    uv.write_text(uv.read_text() + '[ "${5-}" != mcpref.clips.run ] || exit 1\n')
    result = spark.run()
    assert result.returncode != 0 and result.events == [TEXTURES, COLLECT]
