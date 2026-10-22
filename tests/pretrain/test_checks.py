"""The checks (R) of specs/pretrain.md through `pretrain`: each check, its message (paths as
written in the configs) and the order of the checks; nothing written before (R) 6; the
inputs bar of (R) 5; without pairs."""

import shutil

import pytest
from progress_bars import Bar, screen

from .index_world import drop, edit, setting
from .loop_configs import GROUPS
from .loop_modules import NAMES, fakes
from .loop_run import Loop, aborted
from .loop_world import NO_PAIRS, VALIDATION_PAIRS

X, OUT = "w/index", "w/pretrain"
INPUTS = Bar("pretrain inputs", 6, 6, "shard")  # C, Q_c, Q_l: two complete shards each


def index_edit(loop: Loop, change) -> None:
    edit(loop.repo / X / "run.json", change)


def run_aborts(loop: Loop, message: str, modules=None, err: list | None = None) -> None:
    """The run raises RuntimeError(message); the screen above it is `err` (default:
    nothing); nothing written to {out_dir} unless err shows the inputs bar."""
    o = loop.run(modules, raises=RuntimeError)
    aborted(o, message)
    assert screen(o.err) == (err or [])
    if not err:
        assert not loop.out.exists()


def test_index_missing(tmp_path):
    loop = Loop(tmp_path).write()
    (loop.repo / X / "run.json").unlink()
    run_aborts(loop, f"{X} is not complete")


@pytest.mark.parametrize("change", [drop("complete"), setting("complete", False)])
def test_index_not_complete(tmp_path, change):
    loop = Loop(tmp_path).write()
    index_edit(loop, change)
    run_aborts(loop, f"{X} is not complete")


def test_index_as_written(tmp_path):
    """index_dir `w/index/`: the message shows it as written."""
    loop = Loop(tmp_path, index_dir="w/index/").write()
    index_edit(loop, drop("complete"))
    run_aborts(loop, "w/index/ is not complete")


def test_no_training_item(tmp_path):
    """N = counts.train.clips + counts.train.pairs = 0."""
    loop = Loop(tmp_path).write()
    index_edit(loop, setting("counts.train", {"clips": 0, "pairs": 0, "episodes": 0}))
    run_aborts(loop, f"{X} has no training item")


def test_complete_checked_before_items(tmp_path):
    loop = Loop(tmp_path).write()
    index_edit(loop, setting("counts.train", {"clips": 0, "pairs": 0, "episodes": 0}))
    index_edit(loop, drop("complete"))
    run_aborts(loop, f"{X} is not complete")


ALL = str(list(NAMES))
FOUR = str(list(NAMES[:4]))
MODULES = {  # world, given names in order, required names
    "pairs, P missing": (None, ["phi", "phi_loss", "pi", "pi_loss"], ALL),
    "pairs, given in another order": (
        None,
        ["pi", "phi", "P_loss", "phi_loss", "pi_loss"],
        ALL,
    ),
    "pairs, an extra module": (None, [*NAMES, "critic"], ALL),
    "no pairs, P given": (NO_PAIRS, list(NAMES), FOUR),
    "pairs only in validation, P given": (VALIDATION_PAIRS, list(NAMES), FOUR),
}


@pytest.mark.parametrize("world, given, required", MODULES.values(), ids=MODULES)
def test_module_names(tmp_path, world, given, required):
    loop = Loop(tmp_path, **({"world": world} if world else {})).write()
    every = fakes() | {"critic": fakes()["pi"]}
    mods = {k: every[k] for k in given}
    run_aborts(loop, f"{X} needs the modules {required}, given {given}", mods)


def test_items_checked_before_modules(tmp_path):
    loop = Loop(tmp_path).write()
    index_edit(loop, setting("counts.train", {"clips": 0, "pairs": 0, "episodes": 0}))
    run_aborts(loop, f"{X} has no training item", fakes(pairs=False))


def test_module_order_is_not_the_dict_order(tmp_path):
    """The modules given in reverse order: the same run (groups in module order)."""
    a = Loop(tmp_path / "a").write()
    a.run()
    b = Loop(tmp_path / "b").write()
    b.run(dict(reversed(fakes().items())))
    for j in range(4):
        assert (a.checkpoint_dir(j) / "losses.json").read_bytes() == (
            b.checkpoint_dir(j) / "losses.json"
        ).read_bytes()
    ma, mb = a.model(), b.model()
    assert all(ma[k][n].equal(mb[k][n]) for k in ma for n in ma[k])


def with_groups(**declared) -> dict:
    """The fakes with `groups()` of some modules replaced: name -> function of the modules
    giving its groups."""
    mods = fakes()

    def bound(groups):
        return lambda: groups(mods)

    for name, groups in declared.items():
        mods[name].groups = bound(groups)
    return mods


def pi_groups(extra: list) -> callable:
    return lambda m: {"pi": [m["pi"].Wa, m["pi"].ba, m["pi"].V, *extra(m)]}


GROUP_CASES = {
    "a name declared by two modules": (
        {"pi": lambda m: {"phi_c": [m["pi"].Wa, m["pi"].ba, m["pi"].V]}},
        "group phi_c declared twice",
    ),
    "a parameter twice in a group": (
        {"pi": pi_groups(lambda m: [m["pi"].ba])},
        "a parameter is twice in the group pi",
    ),
    "a parameter in two groups of two modules": (
        {"pi_loss": lambda m: {"idle": [m["pi"].V, m["pi_loss"].k]}},
        "a parameter is in the groups pi and idle",
    ),
    "a parameter in no group": (
        {"pi": lambda m: {"pi": [m["pi"].Wa, m["pi"].ba]}},
        "pi has parameters outside the groups",
    ),
    "the first module with a parameter in no group": (
        {"P": lambda m: {"pref": [m["P"].u]}, "phi_loss": lambda m: {"dual": []}},
        "phi_loss has parameters outside the groups",
    ),
    "a parameter twice before one in two groups": (
        {
            "pi": pi_groups(lambda m: [m["pi"].V]),
            "pi_loss": lambda m: {"idle": [m["phi"].A, m["pi_loss"].k]},
        },
        "a parameter is twice in the group pi",
    ),
    "a parameter in two groups before one twice": (
        {"pi": pi_groups(lambda m: [m["phi"].c, m["pi"].V])},
        "a parameter is in the groups phi_c and pi",
    ),
    "declared twice before a repeated parameter": (
        {"pi": lambda m: {"dual": [m["pi"].Wa, m["pi"].Wa]}},
        "group dual declared twice",
    ),
    "repeated before outside": (
        {"pi": lambda m: {"pi": [m["pi"].Wa, m["pi"].Wa]}},
        "a parameter is twice in the group pi",
    ),
}


@pytest.mark.parametrize("declared, message", GROUP_CASES.values(), ids=GROUP_CASES)
def test_groups(tmp_path, declared, message):
    loop = Loop(tmp_path).write()
    run_aborts(loop, message, with_groups(**declared))


def test_parameter_shared_by_two_modules(tmp_path):
    """phi's A also a parameter of pi, declared by both: in the groups phi_A and pi."""
    loop = Loop(tmp_path).write()
    mods = fakes()
    mods["pi"].shared = mods["phi"].A
    mods["pi"].groups = lambda: {"pi": [mods["pi"].Wa, mods["pi"].shared,
                                        mods["pi"].ba, mods["pi"].V]}  # fmt: skip
    run_aborts(loop, "a parameter is in the groups phi_A and pi", mods)


DECLARED = "['phi_A', 'phi_c', 'dual', 'pi', 'idle', 'pref']"
CONFIGS = {
    "a group missing": (
        {k: v for k, v in GROUPS.items() if k != "idle"},
        "['pref', 'pi', 'dual', 'phi_c', 'phi_A']",
    ),
    "an extra group": (
        GROUPS | {"aux": {"lr": 0.1, "weight_decay": 0.0}},
        "['pref', 'pi', 'idle', 'dual', 'phi_c', 'phi_A', 'aux']",
    ),
}


@pytest.mark.parametrize("groups, configured", CONFIGS.values(), ids=CONFIGS)
def test_groups_of_the_config(tmp_path, groups, configured):
    """The declared names (group order) and the config's (config order) as sets."""
    loop = Loop(tmp_path, groups=groups).write()
    message = f"the modules declare the groups {DECLARED}, the config {configured}"
    run_aborts(loop, message)


def test_outside_before_the_config(tmp_path):
    loop = Loop(tmp_path, groups={"phi_A": GROUPS["phi_A"]}).write()
    mods = with_groups(pi=lambda m: {"pi": [m["pi"].Wa]})
    run_aborts(loop, "pi has parameters outside the groups", mods)


def test_modules_before_groups(tmp_path):
    loop = Loop(tmp_path).write()
    mods = with_groups(pi=lambda m: {"pi": [m["pi"].Wa, m["pi"].Wa]})
    del mods["P_loss"]
    message = f"{X} needs the modules {ALL}, given {list(NAMES[:5])}"
    run_aborts(loop, message, mods)


def inputs_edit(loop: Loop, directory: str, change) -> None:
    edit(loop.repo / directory / "run.json", change)


@pytest.mark.parametrize("directory", ["w/clips", "w/pair_clips"])
def test_clips_not_complete(tmp_path, directory):
    """C or Q_c without its `clips` marker: aborts before the inputs bar."""
    loop = Loop(tmp_path).write()
    inputs_edit(loop, directory, drop("clips"))
    run_aborts(loop, f"{directory} is not complete")


def test_clips_as_written_in_the_index(tmp_path):
    """C as written in X's run.json (`w/clips/`)."""
    loop = Loop(tmp_path).write()
    index_edit(loop, setting("config.clips_dir", "w/clips/"))
    inputs_edit(loop, "w/clips", setting("clips", None))
    run_aborts(loop, "w/clips/ is not complete")


def test_groups_before_clips(tmp_path):
    loop = Loop(tmp_path).write()
    inputs_edit(loop, "w/clips", drop("clips"))
    mods = with_groups(pi=lambda m: {"pi": [m["pi"].Wa]})
    run_aborts(loop, "pi has parameters outside the groups", mods)


def test_clips_before_pair_clips(tmp_path):
    loop = Loop(tmp_path).write()
    inputs_edit(loop, "w/pair_clips", drop("clips"))
    inputs_edit(loop, "w/clips", drop("clips"))
    run_aborts(loop, "w/clips is not complete")


def test_completeness_before_fingerprints(tmp_path):
    """Q_c not complete and C's records changed: Q_c's message, no bar."""
    loop = Loop(tmp_path).write()
    records = loop.repo / "w/clips/shards/000000/records.json"
    records.write_bytes(records.read_bytes() + b" ")
    inputs_edit(loop, "w/pair_clips", drop("clips"))
    run_aborts(loop, "w/pair_clips is not complete")


def test_pair_clips_unchecked_without_pairs(tmp_path):
    """No pairs in the index: Q_c and Q_l are not read (absent here)."""
    loop = Loop(tmp_path, world=NO_PAIRS).write()
    assert not (loop.repo / "w/pair_clips").exists()
    o = loop.run()
    assert screen(o.err)[0] == Bar("pretrain inputs", 2, 2, "shard")


def append_space(path) -> None:
    path.write_bytes(path.read_bytes() + b" ")


CHANGES = {  # a change of an input after X was built -> the fingerprint named
    "C's records": (
        lambda r: append_space(r / "w/clips/shards/000001/records.json"),
        "clips",
    ),
    "C's classifier": (
        lambda r: edit(r / "w/clips/run.json", setting("classifier", "e")),
        "clips",
    ),
    "Q_c's records": (
        lambda r: append_space(r / "w/pair_clips/shards/000000/records.json"),
        "pair_clips",
    ),
    "Q_c's states": (
        lambda r: edit(r / "w/pair_clips/run.json", setting("config.width", 5)),
        "pair_clips",
    ),
    "Q_l's pairs": (
        lambda r: append_space(r / "w/pair_labels/shards/000001/pairs.json"),
        "pair_labels",
    ),
}


@pytest.mark.parametrize("change, name", CHANGES.values(), ids=CHANGES)
def test_fingerprint_changed(tmp_path, change, name):
    """After the inputs bar: `{name} changed since {X} was built`; nothing written."""
    loop = Loop(tmp_path).write()
    change(loop.repo)
    run_aborts(loop, f"{name} changed since {X} was built", err=[INPUTS])
    assert not loop.out.exists()


def test_fingerprints_in_order(tmp_path):
    loop = Loop(tmp_path).write()
    for change, _ in CHANGES.values():
        change(loop.repo)
    run_aborts(loop, f"clips changed since {X} was built", err=[INPUTS])


def test_pair_clips_before_pair_labels(tmp_path):
    loop = Loop(tmp_path).write()
    CHANGES["Q_l's pairs"][0](loop.repo)
    CHANGES["Q_c's records"][0](loop.repo)
    run_aborts(loop, f"pair_clips changed since {X} was built", err=[INPUTS])


def test_free_keys_keep_the_fingerprints(tmp_path):
    """Keys of C's run.json outside the clip-defining fields, a staged shard: no abort."""
    loop = Loop(tmp_path).write()
    inputs_edit(loop, "w/clips", setting("config.workers", 9))
    inputs_edit(loop, "w/pair_clips", setting("note", "other"))
    append_space(loop.repo / "w/clips/shards/000002.part/records.json")
    assert loop.run().error is None


def test_inputs_nothing_to_do(tmp_path):
    """C's shards gone (no pairs): the line `pretrain inputs: nothing to do`, then C's
    fingerprint differs."""
    loop = Loop(tmp_path, world=NO_PAIRS).write()
    shutil.rmtree(loop.repo / "w/clips/shards")
    message = f"clips changed since {X} was built"
    run_aborts(loop, message, err=["pretrain inputs: nothing to do"])


def test_config_changed(tmp_path):
    """A rerun with other config bytes: aborts after the inputs, run.json unchanged."""
    loop = Loop(tmp_path).write()
    loop.run()
    before = (loop.out / "run.json").read_bytes()
    loop.path.write_text(loop.path.read_text() + "# a comment\n")
    o = loop.run(raises=RuntimeError)
    aborted(o, f"config changed since the run started; delete {OUT}")
    assert screen(o.err) == [INPUTS]
    assert (loop.out / "run.json").read_bytes() == before


def test_index_changed(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    index_edit(loop, setting("note", 1))
    o = loop.run(raises=RuntimeError)
    aborted(o, f"index changed since the run started; delete {OUT}")


def test_config_before_index(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    index_edit(loop, setting("note", 1))
    loop.path.write_text(loop.path.read_text() + "# a comment\n")
    aborted(loop.run(raises=RuntimeError), f"config changed since the run started; delete {OUT}")  # fmt: skip


def test_out_dir_as_written(tmp_path):
    loop = Loop(tmp_path, out_dir="w/pretrain/").write()
    loop.run()
    loop.path.write_text(loop.path.read_text() + "# a comment\n")
    message = "config changed since the run started; delete w/pretrain/"
    aborted(loop.run(raises=RuntimeError), message)


def test_inputs_before_rerun_check(tmp_path):
    loop = Loop(tmp_path).write()
    loop.run()
    loop.path.write_text(loop.path.read_text() + "# a comment\n")
    CHANGES["C's records"][0](loop.repo)
    aborted(loop.run(raises=RuntimeError), f"clips changed since {X} was built")
