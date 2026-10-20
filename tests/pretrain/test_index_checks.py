"""(R) of specs/train-index.md, the checks before the inputs: each check, its message
(paths as written), their order, none of Q_c and Q_l without pairs; no bar drawn and
nothing written to {out_dir} by an abort."""

import pytest

from .index_oracle import CLIP_FIELDS, sha
from .index_run import aborted
from .index_world import NO_PAIRS, drop, edit, setting

CHECKS = {  # name: ((file under the repository root, change), ...), message
    "C without run.json": ([("w/clips/run.json", None)], "w/clips is not complete"),
    "C without clips": ([("w/clips/run.json", drop("clips"))], "w/clips is not complete"),
    "Q_c without run.json": (
        [("w/pair_clips/run.json", None)],
        "w/pair_clips is not complete",
    ),
    "Q_c without clips": (
        [("w/pair_clips/run.json", drop("clips"))],
        "w/pair_clips is not complete",
    ),
    "C lacks two": (
        [("w/clips/run.json", drop("sha256.overlay_screens")),
         ("w/clips/run.json", drop("classifier"))],
        "w/clips lacks ['sha256.overlay_screens', 'classifier']",
    ),
    "C lacks sha256": (
        [("w/clips/run.json", drop("sha256"))],
        f"w/clips lacks {list(CLIP_FIELDS[3:9])}",
    ),
    "Q_c lacks one": (
        [("w/pair_clips/run.json", drop("config.states"))],
        "w/pair_clips lacks ['config.states']",
    ),
    "differs as JSON text": (
        [("w/pair_clips/run.json", setting("config.width", 640.0))],
        "w/pair_clips differs from w/clips in ['config.width']",
    ),
    "differs in two": (
        [("w/pair_clips/run.json", setting("classifier", "d:other")),
         ("w/pair_clips/run.json", setting("config.states", 17))],
        "w/pair_clips differs from w/clips in ['config.states', 'classifier']",
    ),
    "differs in a hash": (
        [("w/clips/run.json", setting("sha256.gui_actions", sha(b"x")))],
        "w/pair_clips differs from w/clips in ['sha256.gui_actions']",
    ),
    "Q_l without run.json": (
        [("w/pair_labels/run.json", None)],
        "w/pair_labels is not complete",
    ),
    "Q_l complete false": (
        [("w/pair_labels/run.json", setting("complete", False))],
        "w/pair_labels is not complete",
    ),
    "Q_l without complete": (
        [("w/pair_labels/run.json", drop("complete"))],
        "w/pair_labels is not complete",
    ),
}  # fmt: skip


def apply(site, changes) -> None:
    for rel, change in changes:
        path = site.repo / rel
        if change is None:
            path.unlink()
        else:
            edit(path, change)


@pytest.mark.parametrize("check", CHECKS.values(), ids=CHECKS)
def test_check_before_the_inputs(make_site, check):
    """(R): the message; no bar drawn above it (before (I)); {out_dir} not created."""
    site = make_site()
    apply(site, check[0])
    assert aborted(site.run(), check[1]) == []
    assert not site.out.exists()


ORDER = {  # name: (checks applied together, the one reported)
    "1 C before 1 Q_c": (["C without clips", "Q_c without run.json"], "C without clips"),
    "1 Q_c before 2 C": (["C lacks two", "Q_c without clips"], "Q_c without clips"),
    "2 C before 2 Q_c": (["Q_c lacks one", "C lacks sha256"], "C lacks sha256"),
    "2 Q_c before 3": (["differs in two", "Q_c lacks one"], "Q_c lacks one"),
    "3 before 4": (["Q_l without run.json", "differs as JSON text"], "differs as JSON text"),
    "1 before 4": (["Q_l complete false", "C without run.json"], "C without run.json"),
}  # fmt: skip


@pytest.mark.parametrize("names", ORDER.values(), ids=ORDER)
def test_check_order(make_site, names):
    """(R) in the order 1 (C, then Q_c), 2 (C, then Q_c), 3, 4: the first failing one."""
    together, first = names
    site = make_site()
    for n in together:
        apply(site, CHECKS[n][0])
    assert aborted(site.run(), CHECKS[first][1]) == []


def test_without_pairs_q_never_checked(make_site):
    """Pair dirs null: Q_c and Q_l are not read; their checks do not apply."""
    site = make_site(**NO_PAIRS)
    (site.repo / "w/pair_clips").mkdir(parents=True)
    (site.repo / "w/pair_clips/run.json").write_text("not JSON")
    r = site.run()
    assert r.code == 0, r.err


def test_without_pairs_checks_of_c(make_site):
    """Pair dirs null: the checks of C still apply."""
    site = make_site(**NO_PAIRS)
    edit(site.repo / "w/clips/run.json", drop("config.height"))
    assert aborted(site.run(), "w/clips lacks ['config.height']") == []


def test_paths_as_written(make_site):
    """Messages show the config's path strings as written, not normalised."""
    site = make_site(write=False)
    site.config |= {"clips_dir": "./w//clips/", "pair_clips_dir": "w/pair_clips/"}
    site.write()
    edit(site.repo / "w/pair_clips/run.json", setting("config.states", 3))
    assert aborted(site.run(), "w/pair_clips/ differs from ./w//clips/ in ['config.states']") == []  # fmt: skip
