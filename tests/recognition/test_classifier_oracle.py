"""gamma of src/mcpref/recognition/classifier.py against specs/gui-classifier.md (1), (2) on
random fake gui-masks outputs (classifier_world.py) with values near 0 and 255: a naive
transcription of gui-masks (4), (5) on whole bool masks (masks_oracle.py); each frame's
label independent of its batch."""

from fractions import Fraction

import numpy as np
import pytest
from classifier_world import Masks, both, load
from masks_oracle import predict, score

W, H = 6, 4
pytestmark = pytest.mark.usefixtures("away")

VALUES = [0, 1, 2, 3, 4, 250, 251, 252, 253, 254, 255]
RANDOM = {"inv": ["m1", "m2", "m3", "m4"], "chest": ["main"], "empty": ["x"]}
RANDOM |= {"craft": ["main", "b"], "beacon": ["only"]}
NO_PAIR = {("inv", "m2"), ("empty", "x")}
THRESHOLDS = [(0, "0", "0"), (1, "0.5", "0.1"), (2, "0.75", "0"), (4, "0.5", "0.25")]
THRESHOLDS += [(250, "0.9", "0.1"), (253, "0.25", "0")]


def random_world(rng) -> dict:
    """Random (d) pairs with values near 0 and 255, non-empty random masks."""
    out = {}
    for k, members in RANDOM.items():
        out[k] = {}
        for m in members:
            mask = rng.random((H, W)) < 0.5
            mask.flat[rng.integers(W * H)] = True
            pair = (rng.choice(VALUES, (H, W, 3)).astype(np.uint8), mask)
            out[k][m] = None if (k, m) in NO_PAIR else pair
    return out


def random_frames(rng, classes: dict, n: int) -> list:
    """Templates with some pixels replaced, and random frames."""
    templates = [p[0] for ms in classes.values() for p in ms.values() if p is not None]
    frames = []
    for i in range(n):
        x = templates[i % len(templates)].copy()
        if i % 5 == 4:
            x = rng.choice(VALUES, x.shape).astype(np.uint8)
        hit = rng.random(x.shape[:2]) < rng.choice([0.05, 0.2, 0.5])
        x[hit] = rng.choice(VALUES, (int(hit.sum()), 3))
        frames.append(x)
    return frames


@pytest.mark.parametrize("tau, theta, mu", THRESHOLDS)
def test_against_naive_transcription(tmp_path, repo, tau, theta, mu):
    """(1): gui-masks (4) for (d) on whole bool masks, (5) with theta, mu, order K."""
    rng = np.random.default_rng(tau)
    classes = random_world(rng)
    frames = random_frames(rng, classes, 60)
    masks = Masks(tmp_path, repo, classes, both(tau, theta, mu), W, H).write()
    exact = Fraction(theta), Fraction(mu)
    want = [
        predict(
            {k: score(x, ps, tau) for k, ps in masks.pairs().items()},
            list(classes),
            *exact,
        )
        for x in frames
    ]
    assert len(set(want)) >= 2, want  # the case discriminates
    assert load(masks)[0](frames) == want


def test_batch_independence(tmp_path, repo):
    """(2): each frame's label is that of the frame alone, whatever its batch."""
    rng = np.random.default_rng(7)
    classes = random_world(rng)
    frames = random_frames(rng, classes, 30)
    g = load(Masks(tmp_path, repo, classes, both(2, "0.5", "0.1"), W, H).write())[0]
    alone = [g([x])[0] for x in frames]
    assert len(set(alone)) >= 2, alone
    assert g(frames) == alone
    assert g(frames[::-1]) == alone[::-1]
    assert g(frames[3:7] + frames[:3]) == alone[3:7] + alone[:3]
    assert g([frames[0]] * 3 + frames[5:6]) == [alone[0]] * 3 + alone[5:6]


def test_frames_in_any_memory_layout(tmp_path, repo):
    """(1): a frame is its H x W x 3 values (Domain), whatever its strides: strided and
    Fortran-ordered views label as their contiguous copies."""
    rng = np.random.default_rng(8)
    classes = random_world(rng)
    frames = random_frames(rng, classes, 20)
    g = load(Masks(tmp_path, repo, classes, both(2, "0.5", "0.1"), W, H).write())[0]
    big = np.zeros((len(frames), 2 * H, 2 * W, 3), np.uint8)
    big[:, ::2, ::2] = frames
    views = [*big[:, ::2, ::2], *(np.asfortranarray(x) for x in frames)]
    assert not any(v.flags.c_contiguous for v in views)
    assert g(views) == g(frames) * 2
