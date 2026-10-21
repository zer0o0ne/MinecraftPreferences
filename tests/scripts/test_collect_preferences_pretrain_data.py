"""scripts/collect_preferences_pretrain_data.sh against specs/preference-pool.md (Interface:
textures, then `mcpref.pool.run`; blocks vlm-labels and acyclic append their lines) and the
script preamble of specs/spark-env.md, with a fake uv."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
TEXTURES = (*RUN, "mcpref.gui.textures", "configs/gui-screens.yaml")
POOL = (*RUN, "mcpref.pool.run", "configs/preference-pool.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="collect_preferences_pretrain_data.sh")


@pytest.mark.parametrize(
    "where, argv",
    [("cwd", None), ("repo/scripts", ["./collect_preferences_pretrain_data.sh"])],
)
def test_textures_then_the_pool_from_repository_root(spark, where, argv):
    """The textures, then the pool, first, all from the repository root."""
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events[:2] == [TEXTURES, POOL]
    assert all(e[:5] == RUN for e in result.events)
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_textures_stop_the_script(spark):
    spark.fail = "run"
    result = spark.run()
    assert result.returncode != 0 and result.events == [TEXTURES]


def test_failing_pool_stops_the_script(spark):
    """The pool fails (set -e of the preamble): nothing runs after it."""
    uv = spark.bin / "uv"
    uv.write_text(uv.read_text() + '[ "${5-}" != mcpref.pool.run ] || exit 1\n')
    result = spark.run()
    assert result.returncode != 0 and result.events == [TEXTURES, POOL]
