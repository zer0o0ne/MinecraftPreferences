"""`main` of specs/acyclic.md (Interface): exit 0; exit 1 with
`failed("mcpref.preferences.acyclic", error)` as the last line of stderr for the exceptions
of `FAILURES` (a config `ValueError`, an unreadable config, a check, `complete_shards`
raising with its resolved path); others propagate; `argparse`; `python -m`."""

import os
import subprocess
import sys

import pytest
import yaml
from pretrain.index_run import interrupting
from progress_bars import assert_aborted

from mcpref.preferences.acyclic import main

from .acyclic_site import CONFIG, PREFIX, make_site


def last_line(r) -> str:
    assert_aborted(r.code, r.err)
    return r.err.splitlines()[-1]


def test_exit_0(tmp_path):
    """Inputs resolved against the repository root, not the working directory: exit 0,
    nothing on stdout."""
    r = make_site(tmp_path).run()
    assert (r.code, r.stdout) == (0, ""), r.err


HUGE = 10**400  # a YAML integer beyond any float


@pytest.mark.parametrize(
    "change",
    [
        {"refusal_share": 1},
        {"refusal_share": HUGE},
        {"weights": {"X": HUGE, "R": None}},
    ],
    ids=["refusal_share 1", "huge refusal_share", "huge weight"],
)
def test_config_error(tmp_path, change):
    """A violated constraint (a huge integer included): exit 1, `ValueError` last as the
    only line; nothing written."""
    site = make_site(tmp_path, **change)
    r = site.run()
    assert assert_aborted(r.code, r.err) == []
    assert last_line(r).startswith(f"{PREFIX}ValueError: ")
    assert not site.out.exists()


def test_missing_config_file(tmp_path):
    """An unreadable config file (OSError): exit 1."""
    site = make_site(tmp_path, write=False)
    r = site.run(argv=[str(tmp_path / "absent.yaml")])
    assert last_line(r).startswith(f"{PREFIX}FileNotFoundError: ")


def test_complete_shards_message_with_resolved_path(tmp_path):
    """A foreign entry in {V_l}/shards: `complete_shards` raises with the resolved path."""
    site = make_site(tmp_path)
    (site.votes / "shards" / "notes.txt").write_text("")
    message = f"{PREFIX}ValueError: {site.votes / 'shards'}: foreign entry notes.txt"
    assert last_line(site.run()) == message


def test_exception_outside_failures_propagates(tmp_path):
    """An exception not in FAILURES (ArithmeticError) is not turned into exit 1."""
    site = make_site(tmp_path)
    with pytest.raises(ArithmeticError):
        site.run(interrupting("correction shards", 1, error=ArithmeticError))


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    """One positional config path; otherwise argparse exits 2 with its usage."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2 and "usage:" in capsys.readouterr().err


def test_python_m(tmp_path):
    """`python -m mcpref.preferences.acyclic <config>` runs main against the repository
    root: V_l absent there -> exit 1 with its message, the path as written."""
    missing = "data/__no_such_acyclic_input__"
    assert not os.path.exists(os.path.join(os.path.dirname(__file__), "../..", missing))
    path = tmp_path / "c.yaml"
    config = {
        **CONFIG,
        "votes_dir": missing,
        "out_dir": "data/__no_such_acyclic_output__",
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TQDM_")}
    proc = subprocess.run(
        [sys.executable, "-m", "mcpref.preferences.acyclic", str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 1 and proc.stdout == "", proc.stderr
    assert (
        proc.stderr.splitlines()[-1]
        == f"{PREFIX}RuntimeError: {missing} is not complete"
    )
