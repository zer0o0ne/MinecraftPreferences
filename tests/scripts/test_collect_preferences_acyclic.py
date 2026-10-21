"""The line of specs/acyclic.md (Interface: Script) in
scripts/collect_preferences_pretrain_data.sh, with a fake uv: `main` of
mcpref.preferences.acyclic right after the line of vlm-labels, the script's last, from the
repository root; a failing vlm-labels stops the script before it; its failure fails the
script."""

import os

import pytest

CFG = {"driver_min": "575.64.05", "python_version": "3.10.18", "uv_version": "0.13.2"}
RUN = (f"uv@{CFG['uv_version']}", "run", "--locked", "python", "-m")
LABELS = (*RUN, "mcpref.preferences.labels", "configs/vlm-labels.yaml")
ACYCLIC = (*RUN, "mcpref.preferences.acyclic", "configs/acyclic.yaml")


@pytest.fixture
def spark(make_spark):
    return make_spark(CFG, script="collect_preferences_pretrain_data.sh")


def failing(spark, module: str) -> None:
    """The fake uv fails when it runs `module`."""
    uv = spark.bin / "uv"
    uv.write_text(uv.read_text() + f'[ "${{5-}}" != {module} ] || exit 1\n')


@pytest.mark.parametrize(
    "where, argv",
    [("cwd", None), ("repo/scripts", ["./collect_preferences_pretrain_data.sh"])],
)
def test_acyclic_after_the_labels(spark, where, argv):
    result = spark.run(cwd=spark.tmp / where, argv=argv)
    assert result.returncode == 0, result.stderr
    assert result.events.count(ACYCLIC) == 1
    assert result.events[-2:] == [LABELS, ACYCLIC]
    assert {os.path.realpath(c) for c in result.cwds} == {os.path.realpath(spark.repo)}


def test_failing_labels_stop_before_acyclic(spark):
    """vlm-labels fails (set -e of the preamble): the correction does not run."""
    failing(spark, "mcpref.preferences.labels")
    result = spark.run()
    assert result.returncode != 0 and ACYCLIC not in result.events


def test_failing_acyclic_fails_the_script(spark):
    failing(spark, "mcpref.preferences.acyclic")
    result = spark.run()
    assert result.returncode != 0 and result.events[-1] == ACYCLIC
