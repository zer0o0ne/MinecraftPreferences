"""(R) 6 of specs/pretrain.md (revised by specs/policy.md) through `pretrain`: the fingerprints
of the module configs `perception`, `action_mask`, `policy`, `preference` and of the catalogue
named by `screens` of the action-mask config (read by `load_config` of `gui/mask.py`), after
`config` and `index`, in this order, also without pairs; a changed one aborts a rerun with
`{name} changed since the run started; delete {out_dir}` and leaves `run.json` unchanged."""

import itertools

import pytest
from progress_bars import Bar, screen

from .loop_configs import MODULE_CONFIGS, SCREENS
from .loop_run import Loop, aborted
from .loop_world import NO_PAIRS

OUT = "w/pretrain"
FILES = {**MODULE_CONFIGS, "screens": SCREENS}  # name -> path under the repository root


def changed(name: str) -> str:
    return f"{name} changed since the run started; delete {OUT}"


def append(loop: Loop, name: str, text: str = "# changed\n") -> None:
    path = loop.repo / FILES[name]
    path.write_text(path.read_text() + text)


def rerun_aborts(loop: Loop, message: str, inputs: int = 6) -> None:
    """The rerun aborts after the inputs bar; run.json as the first run left it."""
    before = (loop.out / "run.json").read_bytes()
    o = loop.run(raises=RuntimeError)
    aborted(o, message)
    assert screen(o.err) == [Bar("pretrain inputs", inputs, inputs, "shard")]
    assert (loop.out / "run.json").read_bytes() == before


@pytest.mark.parametrize("name", list(FILES))
def test_changed_file(tmp_path, name):
    loop = Loop(tmp_path).write()
    loop.run()
    append(loop, name)
    rerun_aborts(loop, changed(name))


ORDER = ["index", *FILES]


@pytest.mark.parametrize("first, second", list(itertools.pairwise(ORDER)))
def test_order(tmp_path, first, second):
    """Two consecutive fingerprints changed: the earlier one is named."""
    loop = Loop(tmp_path).write()
    loop.run()
    append(loop, second)
    if first == "index":
        (loop.repo / "w/index/run.json").write_text(
            (loop.repo / "w/index/run.json").read_text() + " "
        )
    else:
        append(loop, first)
    rerun_aborts(loop, changed(first))


def test_another_catalogue_named(tmp_path):
    """The action-mask config names another catalogue: action_mask changed."""
    loop = Loop(tmp_path).write()
    loop.run()
    other = "w/configs/other-screens.yaml"
    (loop.repo / other).write_text("# screens\n")
    path = loop.repo / FILES["action_mask"]
    path.write_text(path.read_text().replace(SCREENS, other))
    rerun_aborts(loop, changed("action_mask"))


def test_catalogue_resolved_against_the_repository_root(tmp_path):
    """`screens` of the action-mask config is read under the repository root (not the
    working directory, where a file of other bytes lies)."""
    loop = Loop(tmp_path).write()
    (loop.cwd / SCREENS).parent.mkdir(parents=True)
    (loop.cwd / SCREENS).write_text("# another catalogue\n")
    loop.run()
    append(loop, "screens")
    rerun_aborts(loop, changed("screens"))


@pytest.mark.parametrize("name", ["preference", "screens"])
def test_without_pairs(tmp_path, name):
    """No pair in the index: the fingerprints of every module config still checked."""
    loop = Loop(tmp_path, world=NO_PAIRS).write()
    loop.run()
    append(loop, name)
    rerun_aborts(loop, changed(name), inputs=2)


def test_unchanged_rerun_passes(tmp_path):
    """The same files: the rerun passes the checks (`pretrain: already done`)."""
    loop = Loop(tmp_path).write()
    loop.run()
    o = loop.run()
    assert o.error is None and screen(o.err)[-1] == "pretrain: already done"


def test_action_mask_config_read_by_its_loader(tmp_path):
    """An action-mask config violating its constraints (pitch_limit 0): `ValueError`."""
    loop = Loop(tmp_path).write()
    (loop.repo / FILES["action_mask"]).write_text(
        f'screens: "{SCREENS}"\npitch_limit: 0\n'
    )
    o = loop.run(raises=ValueError)
    assert isinstance(o.error, ValueError)
