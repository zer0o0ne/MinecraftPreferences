"""`main` of specs/vlm-labels.md (Behaviour (R); Interface): the checks of (R) in order,
with their messages, before any bar or output; exit 0; exit 1 with
`failed("mcpref.preferences.labels", error)` as the last line of stderr for the exceptions
of `FAILURES`; others propagate; `argparse`; `python -m mcpref.preferences.labels`."""

import os
import subprocess
import sys

import pytest
import yaml
from progress_bars import assert_aborted

from mcpref.preferences.labels import main

from .config_files import TINY
from .labels_world import HAND, pool_run


def last_line(r) -> str:
    assert_aborted(r.code, r.err)
    return r.err.splitlines()[-1]


def checked(site, message: str) -> None:
    """Exit 1 with the RuntimeError of a check of (R) as the only line on stderr; no
    call, nothing in {out_dir}."""
    r = site.run()
    assert assert_aborted(r.code, r.err) == []
    assert last_line(r) == f"mcpref.preferences.labels: RuntimeError: {message}"
    assert site.fake.calls == [] and not site.out.exists()


def test_pool_without_run_json(make_site):
    site = make_site()
    (site.at("pool_dir") / "run.json").unlink()
    checked(site, "w/pair_clips is not complete")


def test_pool_without_clips(make_site):
    """{Q_c}/run.json without the marker `clips`."""
    run = pool_run(HAND)
    del run["clips"]
    checked(make_site(run=run), "w/pair_clips is not complete")


@pytest.mark.parametrize("frames, stride", [(3, 3), (7, 1), (2, 6)])
def test_frames_and_stride_exceed_the_states(make_site, frames, stride):
    """(k_V - 1) s_V > L - 1 = 5."""
    site = make_site(frames=frames, stride=stride)
    checked(site, "frames and stride exceed the 6 states of w/pair_clips")


def test_checks_in_order(make_site):
    """Both checks failing: the completeness of Q_c first."""
    site = make_site(frames=7, stride=1)
    (site.at("pool_dir") / "run.json").unlink()
    checked(site, "w/pair_clips is not complete")


@pytest.mark.parametrize("change", [{"samples": 0}, {"prefix": "**Answer:** "}])
def test_config_error(make_site, change):
    """A violated constraint: exit 1, `ValueError` last; nothing written."""
    site = make_site(**change)
    assert last_line(site.run()).startswith("mcpref.preferences.labels: ValueError: ")
    assert not site.out.exists()


def test_missing_config_file(make_site, tmp_path):
    """An unreadable config file (OSError): exit 1."""
    site = make_site(write=False)
    r = site.run(argv=[str(tmp_path / "absent.yaml")])
    prefix = "mcpref.preferences.labels: FileNotFoundError: "
    assert last_line(r).startswith(prefix)


def test_foreign_entry_in_the_label_shards(make_site):
    """`complete_shards` of {out_dir}/shards raising: exit 1 with its message (the
    resolved path)."""
    site = make_site()
    (site.out / "shards").mkdir(parents=True)
    (site.out / "shards/notes.txt").write_text("")
    shards = site.out / "shards"
    message = f"ValueError: {shards}: foreign entry notes.txt"
    assert last_line(site.run()) == f"mcpref.preferences.labels: {message}"


def test_exception_outside_failures_propagates(make_site):
    """An exception not in FAILURES (ArithmeticError) is not turned into exit 1."""
    site = make_site()
    site.fake.fail_on, site.fake.error = 1, ArithmeticError
    with pytest.raises(ArithmeticError):
        site.run()


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    """One positional config path; otherwise argparse exits 2 with its usage."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2 and "usage:" in capsys.readouterr().err


def test_python_m(tmp_path):
    """`python -m mcpref.preferences.labels <config>` runs main against the repository
    root: Q_c absent there → exit 1 with its message, the path as written."""
    missing, out = (
        "data/__no_such_vlm_labels_input__",
        "data/__no_such_vlm_labels_out__",
    )
    root = os.path.join(os.path.dirname(__file__), "../..")
    assert not os.path.exists(os.path.join(root, missing))
    config = {**TINY, "pool_dir": missing, "out_dir": out, "vlm": "configs/vlm.yaml"}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TQDM_")}
    proc = subprocess.run(
        [sys.executable, "-m", "mcpref.preferences.labels", str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 1 and proc.stdout == "", proc.stderr
    message = f"mcpref.preferences.labels: RuntimeError: {missing} is not complete"
    assert proc.stderr.splitlines()[-1] == message
    assert not os.path.exists(os.path.join(root, out))
