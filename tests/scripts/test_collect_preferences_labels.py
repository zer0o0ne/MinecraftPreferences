"""The line of specs/vlm-labels.md (Interface: Script) in
scripts/collect_preferences_pretrain_data.sh, with a fake uv: `main` of
mcpref.preferences.labels right after the line of preference-pool, from the repository
root; its failure fails the script."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
POOL = (*RUN, "mcpref.pool.run", "configs/preference-pool.yaml")
LABELS = (*RUN, "mcpref.preferences.labels", "configs/vlm-labels.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="collect_preferences_pretrain_data.sh")


@pytest.mark.parametrize(
    "where, argv",
    [("cwd", None), ("repo/scripts", ["./collect_preferences_pretrain_data.sh"])],
)
def test_labels_after_the_pool(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events.count(LABELS) == 1
    assert result.events.index(LABELS) == result.events.index(POOL) + 1
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_labels_fail_the_script(spark):
    """The labelling fails (set -e of the preamble): the script fails with it."""
    uv = spark.bin / "uv"
    uv.write_text(
        uv.read_text() + '[ "${5-}" != mcpref.preferences.labels ] || exit 1\n'
    )
    result = spark.run()
    assert result.returncode != 0 and result.events[-1] == LABELS
