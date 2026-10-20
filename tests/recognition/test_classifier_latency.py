"""The latency budget (4) of specs/gui-classifier.md on its latency fixture (Tests): the
median over 20 calls, after one warm-up call, of gamma([x]) on one 640 x 360 frame, each call
timed by time.perf_counter, is at most 5 ms under the slow tier's usual parallel load."""

import statistics
import time

import numpy as np
import pytest
from classifier_world import Masks, load

W, H = 640, 360
SIZES = [328, 328, 728, 728, 736, 968, 680, 736, 968, 680, 624, 624, 664, 896, 1168, 896,
         896, 880, 800, 736, 880, 1232, 960, 590, 1200, 880, 592, 1160, 688, 1440, 666, 960,
         960, 640, 680, 568, 1008, 928, 616, 936, 616, 936, 616, 936, 616, 936, 616, 936, 680,
         568, 616, 936, 700, 208, 2235]  # fmt: skip
THRESHOLDS = dict.fromkeys(("agreement", "macro_recall"), (10, "0.5", "0.1"))


def fixture_classes(rng) -> dict:
    """Pair i: M_i = the flat positions 640 * 6i + j, j < |M_i|; random T; one per class."""
    classes = {}
    for i, size in enumerate(SIZES):
        mask = np.zeros(W * H, bool)
        mask[W * 6 * i : W * 6 * i + size] = True
        t = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        classes[f"class{i:02d}"] = {"member": (t, mask.reshape(H, W))}
    return classes


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.usefixtures("away")
def test_latency(tmp_path, repo):
    rng = np.random.default_rng(2026)
    classes = fixture_classes(rng)
    assert len(SIZES) == 55 and sum(SIZES) == 44799  # the Spark run's 44,799 pixels
    masks = Masks(tmp_path, repo, classes, THRESHOLDS, W, H).write()
    gamma, _ = load(masks)
    x = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    assert len(gamma([x])) == 1  # warm-up
    times = []
    for _ in range(20):
        start = time.perf_counter()
        gamma([x])
        times.append(time.perf_counter() - start)
    assert statistics.median(times) <= 0.005, times
