"""(1)-(3) of specs/f3-overlay.md: `overlay_score` and `overlay_shown` of
src/mcpref/recognition/debug_overlay.py on synthetic Y planes, against hand cases and
overlay_oracle.score; masks P, Q of the hand-made font by overlay_oracle.masks."""

import numpy as np
import overlay_oracle as oo
import pytest
import yaml
from drawn_world import hand_font

from mcpref.recognition.debug_overlay import load_config, overlay_score, overlay_shown

W, H = 6, 12
P, Q = oo.masks(
    hand_font(), "b", W, H
)  # |P| = 4 (x = 2, y = 4..7), |Q| = 9 (y 1, 2, 9)
INT = {  # templates with integer bounds
    "text_luma": {"normal": 200, "dimmed": 70},
    "tolerance": {"normal": 10, "dimmed": 5},
    "box_max": {"normal": 150, "dimmed": 60},
}


@pytest.fixture
def cfg(repo, tmp_path):
    """Loads the starting config with `changes`."""

    def load(**changes):
        data = yaml.safe_load((repo / "configs/f3-overlay.yaml").read_text())
        (tmp_path / "f3.yaml").write_text(yaml.safe_dump({**data, **changes}))
        return load_config(tmp_path / "f3.yaml")

    return load


def plane(p_values, q_values, rest: int = 200) -> np.ndarray:
    """Y with P, Q in row-major order; `rest` elsewhere (normal text luma, above every box
    bound: counted if a share left P or Q)."""
    y = np.full((H, W), rest, dtype=np.uint8)
    y[P], y[Q] = p_values, q_values
    return y


def score(y: np.ndarray, cfg) -> float:
    s = overlay_score(y, (P, Q), cfg)
    assert isinstance(s, float)
    return s


@pytest.mark.parametrize(
    "p_values, q_values, s",
    [  # (1) a_m by hand: |Y - lambda| <= delta inclusive; g_m = 1 (Q at 0)
        ([190, 210, 189, 211], [0] * 9, 2 / 4),  # normal 200 +- 10
        ([65, 75, 64, 76], [0] * 9, 2 / 4),  # dimmed 70 +- 5
        # (1) g_m by hand: Y <= beta inclusive; a_m = 1
        ([200] * 4, [150] * 3 + [151] * 6, 3 / 9),
        ([70] * 4, [60] * 4 + [61] * 5, 4 / 9),
    ],
)
def test_shares_bounds_inclusive(cfg, p_values, q_values, s):
    assert score(plane(p_values, q_values), cfg(**INT)) == s


@pytest.mark.parametrize(
    "p_values, q_values, s",
    [  # (1) with the starting values: no rounding of 208.38, 73.7, 160.1, 63.4
        ([185, 232, 184, 233], [0] * 9, 2 / 4),
        ([66, 81, 65, 82], [0] * 9, 2 / 4),
        ([208] * 4, [160] * 2 + [161] * 7, 2 / 9),
        ([73] * 4, [63] * 5 + [64] * 4, 5 / 9),
    ],
)
def test_shares_starting_values(cfg, p_values, q_values, s):
    assert score(plane(p_values, q_values), cfg()) == s


def test_starting_values_select_the_audited_lumas(cfg):
    """[R5] audit: for integer Y the starting values select text luma {185..232} and
    {66..81}, box Y <= 160 and <= 63."""
    c, text, box_n, box_d = cfg(), set(), set(), set()
    for v in range(256):
        if score(plane([v] * 4, [0] * 9), c) == 1:
            text.add(v)
        if score(plane([208] * 4, [v] * 9), c) == 1:
            box_n.add(v)
        if score(plane([73] * 4, [v] * 9), c) == 1:
            box_d.add(v)
    assert text == set(range(185, 233)) | set(range(66, 82))
    assert (box_n, box_d) == (set(range(161)), set(range(64)))


@pytest.mark.parametrize(
    "changes, p_values, q_values, s",
    [  # (1) integer arithmetic: a uint8 difference wraps 4 - 250 and 250 - 4 ... to 10
        ({"text_luma": {"normal": 250, "dimmed": 128}}, [4] * 4, [0] * 9, 0.0),
        ({"text_luma": {"normal": 4, "dimmed": 128}}, [250] * 4, [0] * 9, 0.0),
        # bounds outside 0..255 are not cast to uint8
        ({"box_max": {"normal": 300, "dimmed": -1}}, [200] * 4, [255] * 9, 1.0),
        ({"box_max": {"normal": -1, "dimmed": -1}}, [200] * 4, [0] * 9, 0.0),
        # a fractional box bound, not rounded: 59.6 holds 59, not 60
        ({"box_max": {"normal": 59.6, "dimmed": -1}}, [200] * 4, [59] * 4 + [60] * 5, 4 / 9),
        # a fractional centre and tolerance: 100.5 +- 0.5 holds 100, 101 only
        (
            {"text_luma": {"normal": 100.5, "dimmed": 128}, "tolerance": {"normal": 0.5, "dimmed": 0}},
            [99, 100, 101, 102], [0] * 9, 2 / 4,
        ),
    ],
)  # fmt: skip
def test_shares_integer_arithmetic(cfg, changes, p_values, q_values, s):
    base = {**INT, "tolerance": {"normal": 10, "dimmed": 0}}
    assert score(plane(p_values, q_values), cfg(**{**base, **changes})) == s


@pytest.mark.parametrize(
    "box_max, p_values, q_values, s",
    [  # (2) S = max(min(a_n, g_n), min(a_d, g_d)), each share by hand
        (None, [200, 200, 200, 70], [0] * 7 + [255] * 2, 3 / 4),  # a_n binds
        (None, [200] * 4, [0] * 5 + [100] * 3 + [255], 8 / 9),  # g_n binds
        (None, [70, 70, 70, 200], [0] * 6 + [100] * 3, 6 / 9),  # dimmed wins, g_d binds
        (None, [200, 200, 70, 70], [0] * 4 + [100] * 5, 2 / 4),  # S_n = 1/2 > S_d = 4/9
        # a_n = 1, g_n = 0, a_d = 0, g_d = 1: the min within a template, not across them
        ({"normal": 60, "dimmed": 150}, [200] * 4, [100] * 9, 0.0),
    ],
)
def test_score_hand_cases(cfg, box_max, p_values, q_values, s):
    changes = INT if box_max is None else {**INT, "box_max": box_max}
    assert score(plane(p_values, q_values), cfg(**changes)) == s


def candidates(lam: dict, delta: dict, beta: dict) -> np.ndarray:
    """Integer lumas at and next to every bound of (1), and 0, 255."""
    edges = [lam[m] + s * delta[m] for m in oo.TEMPLATES for s in (-1, 1)]
    edges += list(beta.values())
    values = {int(np.floor(e)) + d for e in edges for d in (-1, 0, 1, 2)} | {0, 255}
    return np.array(sorted(v for v in values if 0 <= v <= 255))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("ints", [False, True])
def test_score_naive_reference(cfg, seed, ints):
    """(1), (2) against overlay_oracle.score on random planes near the bounds."""
    c = cfg(**INT) if ints else cfg()
    lam, delta, beta = (dict(getattr(c, k)) for k in INT)
    p, q = oo.masks(hand_font(), "A~d cb", 40, 16)
    rng = np.random.default_rng(seed)
    expected = []
    for _ in range(8):
        y = rng.choice(candidates(lam, delta, beta), size=(16, 40)).astype(np.uint8)
        expected.append(oo.score(y, p, q, lam, delta, beta))
        assert overlay_score(y, (p, q), c) == expected[-1]
    assert len(set(expected)) > 1


@pytest.mark.parametrize(
    "p_values, q_values, s",
    [
        ([200, 200, 200, 70], [0] * 9, 3 / 4),
        ([200] * 4, [0] * 7 + [255] * 2, 7 / 9),
        ([200] * 4, [0] * 9, 1.0),
    ],
)
def test_shown_at_threshold(cfg, p_values, q_values, s):
    """(3): shown iff S >= theta; at S = theta, and theta one float above S."""
    y = plane(p_values, q_values)
    assert score(y, cfg(**INT)) == s
    for theta, shown in ((s, True), (float(np.nextafter(s, 2)), False)):
        if theta <= 1:
            assert overlay_shown(y, (P, Q), cfg(**INT, threshold=theta)) is shown


def test_not_shown_at_zero(cfg):
    """(3) with the smallest thresholds: S = 0 is never shown, theta > 0."""
    y = plane([0] * 4, [255] * 9)
    assert score(y, cfg(**INT)) == 0
    assert overlay_shown(y, (P, Q), cfg(**INT, threshold=0.001)) is False
