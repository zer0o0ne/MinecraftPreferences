"""`main` of src/mcpref/pretrain/run.py against specs/policy.md (13): the configs loaded in the
order perception, action_mask (with `load_masks`), policy, preference, resolved against the
repository root; then (R) 1; then the checks of the lengths (X as written, pairs iff
`counts.train.pairs` >= 1); then the modules and `pretrain(cfg, path, modules)`; exit 0, or
exit 1 with `failed("mcpref.pretrain.run", error)` as the last line of stderr for the
exceptions of `FAILURES`; others propagate; argparse; `python -m mcpref.pretrain.run`."""

import dataclasses
import os
import subprocess
import sys

import pytest
import torch
import yaml
from progress_bars import assert_aborted

from mcpref.policy.network import Policy
from mcpref.pretrain.config import load_config
from mcpref.pretrain.run import main

from .index_world import edit, setting
from .loop_configs import MODULE_CONFIGS, SCREENS
from .loop_modules import NAMES, D
from .loop_world import LABELED, NO_PAIRS, VALIDATION_PAIRS
from .run_site import POLICY, PREFERENCE, PREFIX, Site, recorder


def last_line(o) -> str:
    assert_aborted(o.code, o.err)
    return o.err.splitlines()[-1]


def test_exit_0_with_the_loop(tmp_path, repo):
    """The modules of (13) (phi and its loss faked) trained by the loop on the configs
    under the repository root: exit 0, nothing on stdout, model.pt of the six modules."""
    site = Site(tmp_path, repo, world=LABELED).write()
    o = site.main(phi_loss=True)
    assert (o.code, o.out) == (0, ""), o.err
    model = site.model()
    assert list(model) == list(NAMES)
    assert list(model["pi"]) == list(Policy(POLICY, D, ("a",)).state_dict())


def test_modules_given_to_the_loop(tmp_path, repo, masks):
    """pretrain(cfg, path, modules): the config at the path, the path as given, the six
    modules (pi over the approved A, D = len(alpha) of the perception config)."""
    site = Site(tmp_path, repo).write()
    calls = []
    o = site.main(loop=recorder(calls))
    assert o.code == 0 and len(calls) == 1, o.err
    cfg, path, modules = calls[0]
    assert cfg == load_config(site.path) and path == site.path
    assert list(modules) == list(NAMES)
    assert len(site.built) == 1 and site.built[0].alpha == (1.0, 2.0)
    want = Policy(POLICY, 2, masks.names).state_dict()
    got = modules["pi"].state_dict()
    assert list(got) == list(want) and all(torch.equal(got[k], want[k]) for k in want)


@pytest.mark.parametrize(
    "world", [NO_PAIRS, VALIDATION_PAIRS], ids=["none", "validation"]
)
def test_without_pairs_in_the_training_part(tmp_path, repo, world):
    """counts.train.pairs = 0: no P, and L_P > L accepted."""
    big = dataclasses.replace(PREFERENCE, history=9)
    site = Site(tmp_path, repo, world=world, preference=big).write()
    calls = []
    assert site.main(loop=recorder(calls)).code == 0
    assert list(calls[0][2]) == list(NAMES[:4])


def remove(site: Site, key: str) -> str:
    """The module config of `key` deleted; its resolved path."""
    path = site.repo / MODULE_CONFIGS[key]
    path.unlink()
    return str(path)


def spoil(site: Site, key: str) -> None:
    """An unknown key added to the module config of `key` (ValueError at load)."""
    path = site.repo / MODULE_CONFIGS[key]
    path.write_text(path.read_text() + "unknown_key: 1\n")


ORDER = {  # the first config missing, a later one invalid -> the first one's error
    "perception before action_mask": ("perception", "action_mask"),
    "action_mask before policy": ("action_mask", "policy"),
    "policy before preference": ("policy", "preference"),
}


@pytest.mark.parametrize("missing, invalid", ORDER.values(), ids=ORDER)
def test_configs_in_order(tmp_path, repo, missing, invalid):
    site = Site(tmp_path, repo).write()
    spoil(site, invalid)
    path = remove(site, missing)
    line = last_line(site.main())
    assert line.startswith(f"{PREFIX}FileNotFoundError: ") and path in line, line
    assert site.built == []


@pytest.mark.parametrize("key", list(MODULE_CONFIGS))
def test_each_config_loaded(tmp_path, repo, key):
    """Each module config read by its block's loader: an unknown key -> ValueError."""
    site = Site(tmp_path, repo).write()
    spoil(site, key)
    assert last_line(site.main()).startswith(f"{PREFIX}ValueError: ")


def test_masks_loaded_before_the_policy_config(tmp_path, repo):
    """load_masks right after the action-mask config: its catalogue missing is reported
    before an invalid policy config."""
    site = Site(tmp_path, repo).write()
    (site.repo / SCREENS).unlink()
    spoil(site, "policy")
    line = last_line(site.main())
    assert line.startswith(f"{PREFIX}FileNotFoundError: ") and SCREENS in line, line


def test_configs_before_the_index(tmp_path, repo):
    site = Site(tmp_path, repo).write()
    edit(site.repo / "w/index/run.json", setting("complete", False))
    path = remove(site, "preference")
    line = last_line(site.main())
    assert line.startswith(f"{PREFIX}FileNotFoundError: ") and path in line, line


def test_index_not_complete(tmp_path, repo):
    """(R) 1 after the configs, before the lengths (L = 2 here)."""
    site = Site(tmp_path, repo).write()
    edit(site.repo / "w/index/run.json", setting("complete", False))
    edit(site.repo / "w/index/run.json", setting("states", 2))
    assert last_line(site.main()) == f"{PREFIX}RuntimeError: w/index is not complete"


LENGTH_CASES = {  # changes of states, policy, preference, index_dir -> the message
    "L = 2": ({"states": 2}, {}, {}, "w/index has clips of 2 states, the policy needs at least 3"),
    "L_pi other": ({}, {"history": 4}, {}, "the policy history 4 differs from the clip length 3 of w/index"),
    "L_P above L": ({}, {}, {"history": 4}, "the preference history 4 exceeds the clip length 3 of w/index"),
    "X as written": ({"states": 2}, {}, {}, "w/index/ has clips of 2 states, the policy needs at least 3"),
}  # fmt: skip


@pytest.mark.parametrize("run, policy, preference, message", LENGTH_CASES.values(), ids=LENGTH_CASES)  # fmt: skip
def test_lengths(tmp_path, repo, run, policy, preference, message):
    """L = `states` of {X}/run.json; nothing built when a check fails."""
    pol = dataclasses.replace(POLICY, **policy)
    pref = dataclasses.replace(PREFERENCE, **preference)
    changes = {"index_dir": "w/index/"} if message.startswith("w/index/") else {}
    site = Site(tmp_path, repo, policy=pol, preference=pref, **changes).write()
    for key, value in run.items():
        edit(site.repo / "w/index/run.json", setting(key, value))
    calls = []
    assert (
        last_line(site.main(loop=recorder(calls))) == f"{PREFIX}RuntimeError: {message}"
    )
    assert calls == [] and site.built == []


def test_exception_outside_failures_propagates(tmp_path, repo):
    """An exception not in FAILURES (ArithmeticError) is not turned into exit 1."""
    site = Site(tmp_path, repo).write()

    def boom(cfg, path, modules):
        raise ArithmeticError("raised by the test")

    with pytest.raises(ArithmeticError):
        site.main(loop=boom)


@pytest.mark.parametrize("argv", [[], ["a.yaml", "b.yaml"], ["--seed", "1"]])
def test_wrong_arguments(argv, capsys):
    """One positional config path; otherwise argparse exits 2 with its usage."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2 and "usage:" in capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_python_m(tmp_path, repo):
    """`python -m mcpref.pretrain.run <config>` against the repository root: a perception
    config absent there -> exit 1 with its error, the path resolved against the root. Slow:
    the import of the module (torch, transformers) alone takes about 4 s."""
    missing = "data/__no_such_pretrain_input__/perception.yaml"
    assert not (repo / missing).exists()
    config = yaml.safe_load((repo / "configs/pretrain.yaml").read_text())
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config | {"perception": missing}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TQDM_")}
    proc = subprocess.run([sys.executable, "-m", "mcpref.pretrain.run", str(path)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=60, check=False)  # fmt: skip
    assert proc.returncode == 1 and proc.stdout == "", proc.stderr
    line = proc.stderr.splitlines()[-1]
    assert (
        line.startswith(f"{PREFIX}FileNotFoundError: ") and str(repo / missing) in line
    )
