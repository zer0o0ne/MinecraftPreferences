"""`main` of src/mcpref/pool/run.py against specs/preference-pool.md (Interface): with a fake
gui-masks output (tests/clips/run_world.py, under the repository root of the test) and a
fake `collect`: the pool's config, perception-data's config, gamma and gamma_id passed to
`collect`; exit 1 with `mcpref.pool.run: <type>: <message>` as the last line of stderr on
any exception; argparse with one positional config path."""

import subprocess
import sys
from pathlib import Path

import pytest
from classifier_world import gamma_id
from clips.run_world import MARK, World
from progress_bars import Bar, assert_aborted, screen, tqdm_defaults

from mcpref.clips.collect import load_config as load_perception
from mcpref.pool import run

from .pool_site import write_config

PREFIX = "mcpref.pool.run: "


class FakeCollect:
    """`collect(cfg, p, classify, classifier_id)`: records its arguments and the labels of
    `frames`, marks stderr, then raises `error` if given."""

    def __init__(self, frames: list, error: BaseException | None = None):
        self.frames, self.error, self.calls = frames, error, []

    def __call__(self, cfg, p, classify, classifier_id) -> None:
        print(MARK, file=sys.stderr)
        self.calls.append((cfg, p, classify(self.frames), classifier_id))
        if self.error is not None:
            raise self.error


@pytest.fixture
def world(root, monkeypatch):
    """The fake gui-masks output and perception-data's config under the root; the pool's
    config naming it."""
    tqdm_defaults(monkeypatch)
    w = World(root, root)
    w.pool = write_config(root / "preference-pool.yaml", perception="perception-data.yaml", out_dir="pool-out")  # fmt: skip
    return w


def main(monkeypatch, capfd, world, error=None, argv=None):
    """main with a fake collect: (exit code, stdout, stderr, fake)."""
    fake = FakeCollect(world.frames, error)
    monkeypatch.setattr(run, "collect", fake)
    code = run.main([str(world.pool)] if argv is None else argv)
    out, err = capfd.readouterr()
    return code, out, err, fake


def test_configs_gamma_and_its_id_passed_to_collect(world, monkeypatch, capfd):
    """load_config, perception-data's load_config of `perception`, load_classifier, then
    collect; exit 0; the bar `classifier` drawn first, closed before collect."""
    code, out, err, fake = main(monkeypatch, capfd, world)
    assert code == 0 and out == "", err
    ((cfg, p, labels, gid),) = fake.calls
    assert cfg == run.load_config(world.pool)
    assert p == load_perception(Path(world.path))
    assert labels == ["inv", "chest", "other"]
    assert gid == gamma_id(world.masks, "macro_recall")
    assert screen(err) == [Bar("classifier", 3, 3, "shard"), MARK]


def test_failed_check_of_the_classifier(world, monkeypatch, capfd):
    """gui-masks without run.json: exit 1, its message last; collect never called."""
    (world.masks.out / "run.json").unlink()
    code, out, err, fake = main(monkeypatch, capfd, world)
    assert assert_aborted(code, err) == [] and out == "" and fake.calls == []
    assert err.splitlines()[-1] == f"{PREFIX}RuntimeError: {world.masks.masks_dir} is not complete"  # fmt: skip


@pytest.mark.parametrize("error, line", [(ZeroDivisionError("fake collect failed"), "ZeroDivisionError: fake collect failed"), (RuntimeError("fake abort"), "RuntimeError: fake abort")])  # fmt: skip
def test_exception_of_collect(world, monkeypatch, capfd, error, line):
    code, _, err, fake = main(monkeypatch, capfd, world, error)
    above = assert_aborted(code, err)
    assert err.splitlines()[-1] == PREFIX + line and len(fake.calls) == 1
    assert above == [Bar("classifier", 3, 3, "shard"), MARK]


def test_other_base_exceptions_propagate(world, monkeypatch, capfd):
    with pytest.raises(KeyboardInterrupt):
        main(monkeypatch, capfd, world, KeyboardInterrupt())


@pytest.mark.parametrize(
    "change", [{"n_clips": 1}, {"tick_slack": 10**400}, {"perception": "no-such.yaml"}, {"perception": "../perception-data.yaml"}]
)  # fmt: skip
def test_config_errors(world, monkeypatch, capfd, change):
    """A rejected pool config (a huge integer tick_slack included) or a missing
    perception-data config: exit 1 with the one-line message, before any bar; collect
    never called."""
    write_config(world.pool, **({"perception": "perception-data.yaml", "out_dir": "pool-out"} | change))  # fmt: skip
    code, _, err, fake = main(monkeypatch, capfd, world)
    assert assert_aborted(code, err) == [] and fake.calls == []
    assert err.startswith(PREFIX) and err.count("\n") == 1
    if "perception" not in change:
        assert err.startswith(f"{PREFIX}ValueError: ")


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1", "a.yaml"]])
def test_wrong_argument_list_exits_2(world, monkeypatch, capfd, argv):
    with pytest.raises(SystemExit) as info:
        main(monkeypatch, capfd, world, argv=argv)
    assert info.value.code == 2
    out, err = capfd.readouterr()
    assert out == "" and err.startswith("usage: ") and "error: " in err


def test_run_as_module(tmp_path):
    """`python -m mcpref.pool.run <config path>`: exit 1 with the message on a missing
    config; exit 2 with the usage without one."""
    argv = [sys.executable, "-m", "mcpref.pool.run"]
    kw = {"cwd": tmp_path, "capture_output": True, "text": True, "timeout": 60}
    missing = subprocess.run([*argv, str(tmp_path / "missing.yaml")], check=False, **kw)
    assert missing.returncode == 1 and missing.stdout == ""
    assert missing.stderr.startswith(PREFIX) and missing.stderr.count("\n") == 1
    bare = subprocess.run(argv, check=False, **kw)
    assert bare.returncode == 2 and bare.stderr.startswith("usage: ")
