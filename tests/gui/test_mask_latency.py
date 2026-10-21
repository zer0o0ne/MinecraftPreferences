"""(5) of specs/action-mask.md: the median latency of `state_mask` on a world state and on a
GUI state of each class of D; target 0.1 ms, failing only above 1 ms; the medians printed."""

import statistics
import time

from mask_oracle import D

from mcpref.gui.mask import state_mask


def median_ms(masks, gui: bool, cls: str | None) -> float:
    """Median over 100 calls after one warm-up call, each timed by time.perf_counter."""
    state_mask(masks, gui, cls, 12.5, 4)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        state_mask(masks, gui, cls, 12.5, 4)
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


def test_state_mask_latency(masks):
    medians = {"world": median_ms(masks, False, None)}
    medians |= {k: median_ms(masks, True, k) for k in D}
    for name, ms in medians.items():
        print(f"state_mask {name}: {ms:.4f} ms")
    assert max(medians.values()) <= 1.0, medians
