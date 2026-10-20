"""`main` of specs/train-index.md (Interface): exit 0; exit 1 with
`failed("mcpref.pretrain.index", error)` as the last line of stderr for the exceptions of
`FAILURES` (a config `ValueError`, an unreadable config, a check, `complete_shards` raising
with its resolved path); others propagate; `argparse`; `python -m mcpref.pretrain.index`."""

import os
import subprocess
import sys

import pytest
import yaml
from progress_bars import assert_aborted

from mcpref.pretrain.index import main

from .index_run import PREFIX, interrupting
from .index_world import CONFIG


def last_line(r) -> str:
    assert_aborted(r.code, r.err)
    return r.err.splitlines()[-1]


def test_exit_0(make_site):
    """A run on inputs resolved against the repository root, not the working directory:
    exit 0, nothing on stdout."""
    site = make_site()
    r = site.run()
    assert (r.code, r.stdout) == (0, ""), r.err


def test_config_error(make_site):
    """A violated constraint: exit 1, `ValueError` last; nothing read or written."""
    site = make_site(items_per_shard=0)
    assert last_line(site.run()).startswith(f"{PREFIX}ValueError: ")
    assert not site.out.exists()


def test_missing_config_file(make_site, tmp_path):
    """An unreadable config file (OSError): exit 1."""
    site = make_site(write=False)
    r = site.run(argv=[str(tmp_path / "absent.yaml")])
    assert last_line(r).startswith(f"{PREFIX}FileNotFoundError: ")


def test_complete_shards_message_with_resolved_path(make_site):
    """A foreign entry in {C}/shards: `complete_shards` raises; its message shows the
    resolved path."""
    site = make_site()
    (site.repo / "w/clips/shards/notes.txt").write_text("")
    shards = site.repo / "w/clips/shards"
    assert (
        last_line(site.run())
        == f"{PREFIX}ValueError: {shards}: foreign entry notes.txt"
    )


def test_exception_outside_failures_propagates(make_site):
    """An exception not in FAILURES (ArithmeticError) is not turned into exit 1."""
    site = make_site()
    with pytest.raises(ArithmeticError):
        site.run(interrupting("index shards", 1, error=ArithmeticError))


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    """One positional config path; otherwise argparse exits 2 with its usage."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2 and "usage:" in capsys.readouterr().err


def test_python_m(tmp_path):
    """`python -m mcpref.pretrain.index <config>` runs main against the repository root:
    C absent there → exit 1 with its message, the path as written."""
    missing = "data/__no_such_train_index_input__"
    assert not os.path.exists(os.path.join(os.path.dirname(__file__), "../..", missing))
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, "clips_dir": missing, "out_dir": "x"}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TQDM_")}
    proc = subprocess.run(
        [sys.executable, "-m", "mcpref.pretrain.index", str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 1 and proc.stdout == "", proc.stderr
    message = f"{PREFIX}RuntimeError: {missing} is not complete"
    assert proc.stderr.splitlines()[-1] == message
