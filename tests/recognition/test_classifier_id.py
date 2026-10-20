"""gamma_id of `load_classifier` (src/mcpref/recognition/classifier.py) against
specs/gui-classifier.md (3) and its Tests item: an in-test transcription (classifier_world.py);
it changes with tau, theta, mu, a template byte inside a mask and the catalogue, not with a
template byte outside every mask, the other variants' arrays, nor the criterion name when the
thresholds are equal."""

import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest
from classifier_world import Masks, gamma_id, load, shard_arrays

W, H = 6, 4
THRESHOLDS = {"agreement": (3, "0.50", "0"), "macro_recall": (5, "0.75", "0.10")}
pytestmark = pytest.mark.usefixtures("away")


def classes() -> dict:
    """Members without (d) keys among those with, a member name shared by two classes, a
    non-ASCII one, a class without a pair."""
    rng = np.random.default_rng(11)

    def pair():
        mask = rng.random((H, W)) < 0.4
        mask[H - 1, W - 1] = True
        return rng.integers(0, 256, (H, W, 3), dtype=np.uint8), mask

    out = {"inv": {"m1": pair(), "m2": None, "m3": pair()}, "empty": {"x": None}}
    return out | {"chest": {"main": pair()}, "craft": {"main": pair(), "é1": pair()}}


def write(root, repo, thresholds=THRESHOLDS, cls=None, note: str = "") -> Masks:
    return Masks(root, repo, cls or classes(), thresholds, W, H, note).write()


def gid(masks: Masks, criterion: str = "agreement") -> str:
    return load(masks, criterion)[1]


@pytest.mark.parametrize("criterion", ["agreement", "macro_recall"])
def test_identifier_against_transcription(tmp_path, repo, criterion):
    """(3): default separators, str of the Fractions ("1/2", "0", "3/4", "1/10"), h_T over
    the (d) pairs in class and member order, h_K of the catalogue file."""
    masks = write(tmp_path, repo)
    got = gid(masks, criterion)
    assert got == gamma_id(masks, criterion)
    doc = json.loads(got)
    assert list(doc) == ["variant", "tau", "theta", "mu", "templates", "catalogue"]
    want = {"agreement": (3, "1/2", "0"), "macro_recall": (5, "3/4", "1/10")}[criterion]
    assert (doc["variant"], doc["tau"], doc["theta"], doc["mu"]) == ("d", *want)


@pytest.mark.parametrize(
    "criterion, entry",
    [("agreement", (4, "0.50", "0")), ("agreement", (3, "0.5000001", "0")),
     ("agreement", (3, "0.50", "0.01")), ("macro_recall", (5, "0.75", "0.1"))],
)  # fmt: skip
def test_identifier_changes_with_thresholds(tmp_path, repo, criterion, entry):
    """(3): a change of tau, theta or mu changes gamma_id; the same fractions written
    otherwise ("0.1" for "0.10") do not."""
    base = write(tmp_path / "base", repo)
    masks = write(tmp_path / "new", repo, THRESHOLDS | {criterion: entry})
    assert gid(masks, criterion) == gamma_id(masks, criterion)
    exact = [
        (t, Fraction(th), Fraction(mu)) for t, th, mu in (THRESHOLDS[criterion], entry)
    ]
    assert (gid(masks, criterion) == gid(base, criterion)) == (exact[0] == exact[1])


def changed(masks: Masks, k: str, m: str, inside: bool, variant: str = "d") -> None:
    """One template byte of member m of class k changed, at a pixel inside or outside its
    (d) mask; variant `a`: in its (a) template instead."""
    _, mask = masks.classes[k][m]
    r, q = map(int, np.argwhere(mask if inside else ~mask)[0])
    s = list(masks.classes).index(k)
    path = masks.out / "templates/catalogue" / f"{s:06d}" / "pairs.npz"
    arrays = shard_arrays(masks.classes[k], W, H)
    arrays[f"{variant}/{m}/T"][r, q, 1] ^= 0x10
    np.savez(path, **arrays)
    if variant == "d":
        masks.classes[k][m] = (arrays[f"d/{m}/T"], mask)


@pytest.mark.parametrize(
    "k, m, inside, variant, differs",
    [("inv", "m3", True, "d", True), ("craft", "é1", True, "d", True),
     ("inv", "m3", False, "d", False), ("craft", "main", False, "d", False),
     ("inv", "m1", True, "a", False), ("chest", "main", True, "e", False)],
)  # fmt: skip
def test_identifier_and_template_bytes(tmp_path, repo, k, m, inside, variant, differs):
    """(3): T_I enters h_T; T outside M and the other variants' arrays do not."""
    masks = write(tmp_path, repo)
    before = gid(masks)
    changed(masks, k, m, inside, variant)
    assert gid(masks) == gamma_id(masks, "agreement")
    assert (gid(masks) != before) == differs


def test_identifier_changes_with_the_catalogue(tmp_path, repo):
    """(3): h_K of the catalogue file: other bytes, same classes, gui-masks run on it."""
    base = write(tmp_path / "base", repo)
    masks = write(tmp_path / "new", repo, note="rewritten")
    assert gid(masks) == gamma_id(masks, "agreement") != gid(base)
    h_k = hashlib.sha256(masks.catalogue.read_bytes()).hexdigest()
    assert json.loads(gid(masks))["catalogue"] == h_k


def test_identifier_and_the_criterion_name(tmp_path, repo):
    """(3): equal thresholds under both criteria: one gamma_id; different ones: two."""
    equal = write(tmp_path / "eq", repo, dict.fromkeys(THRESHOLDS, (3, "0.5", "0")))
    assert gid(equal, "agreement") == gid(equal, "macro_recall")
    other = write(tmp_path / "ne", repo)
    assert gid(other, "agreement") != gid(other, "macro_recall")


def test_identifier_without_pairs(tmp_path, repo):
    """(3): no class, and classes without a pair: h_T of no bytes."""
    empty = hashlib.sha256(b"").hexdigest()
    for name, cls in (("none", {}), ("nopair", {"x": {"m": None}, "y": {"n": None}})):
        masks = Masks(tmp_path / name, repo, cls, THRESHOLDS, W, H).write()
        assert gid(masks) == gamma_id(masks, "agreement")
        assert json.loads(gid(masks))["templates"] == empty
