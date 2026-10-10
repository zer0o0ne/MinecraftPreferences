"""scripts/download_data.sh: the step of specs/gui-screens.md, then `verify` and `run` of
specs/vpt-gui-frames.md (Interface), after the preamble of specs/spark-env.md."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
PYTHON = ("uv@0.13.2", "run", "--locked", "python", "-m")
TEXTURES = (*PYTHON, "mcpref.gui.textures", "configs/gui-screens.yaml")
VERIFY = (*PYTHON, "mcpref.vpt.gui_frames", "verify", "configs/vpt-gui-frames.yaml")
RUN = (*PYTHON, "mcpref.vpt.gui_frames", "run", "configs/vpt-gui-frames.yaml")


@pytest.mark.parametrize(
    "where, argv", [("cwd", None), ("repo/scripts", ["./download_data.sh"])]
)
def test_textures_then_verify_then_run_from_repository_root(make_spark, where, argv):
    spark = make_spark(CFG, "download_data.sh")
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events == [TEXTURES, VERIFY, RUN]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


@pytest.mark.parametrize(
    "failing, done",
    [("mcpref.gui.textures", [TEXTURES]), ("verify", [TEXTURES, VERIFY]),
     ("run", [TEXTURES, VERIFY, RUN])],
)  # fmt: skip
def test_failing_step_stops_the_script(make_spark, failing, done):
    """set -e: a failing `verify` keeps the full run from starting."""
    spark = make_spark(CFG, "download_data.sh")
    uv = spark.bin / "uv"
    uv.write_text(uv.read_text() + f'[ "${{@: -2:1}}" != {failing} ] || exit 7\n')
    result = spark.run()
    assert result.returncode != 0
    assert result.events == done
