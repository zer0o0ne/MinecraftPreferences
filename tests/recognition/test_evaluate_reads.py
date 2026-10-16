"""The frame reads of `main` of specs/gui-masks.md on the `base` world: (2) reads each
training frame of a class of Q only by row blocks (one `rgb_rows` call per frame and block, in
row order, no whole-frame read), (S) reads each frame of F once, the parts in the order train,
validation, test. Reads are seen through `rgb_rows` of mcpref.recognition.pairs (Interface)
and the `open` audit events of .yuv files outside it."""

import inspect
import math
import sys
from pathlib import Path

import masks_world as mw
from masks_files import write
from masks_run import expected_of, rerun

from mcpref.recognition import pairs


class Reads:
    """`rgb_rows` calls (id, r0, r1) and the whole-frame reads (ids) of one run, in order."""

    active = None

    def __init__(self):
        self.calls, self.whole, self.depth = [], [], 0


def audit(event: str, args: tuple) -> None:
    rec = Reads.active
    if rec and event == "open" and str(args[0]).endswith(".yuv") and not rec.depth:
        rec.whole.append(Path(str(args[0])).stem)


sys.addaudithook(audit)  # for the life of the process; idle without an active recorder


def recording(rec: Reads):
    """A patch wrapping `rgb_rows` to record its calls and activating `rec`; a call reading
    every row counts as a whole-frame read."""
    real = pairs.rgb_rows
    signature = inspect.signature(real)

    def rgb_rows(*args, **kwargs):
        a = signature.bind(*args, **kwargs).arguments
        frame, rows = Path(a["path"]).stem, (a["r0"], a["r1"])
        if rows == (0, a["height"]):
            rec.whole.append(frame)
        else:
            rec.calls.append((frame, *rows))
        rec.depth += 1
        try:
            return real(*args, **kwargs)
        finally:
            rec.depth -= 1

    def patch(mp) -> None:
        mp.setattr(pairs, "rgb_rows", rgb_rows)
        mp.setattr(Reads, "active", rec)

    return patch


def test_reads(tmp_path, repo):
    exp = expected_of("base")
    write(tmp_path, repo, mw.frames(), mw.CONFIG)
    rec = Reads()
    assert rerun(tmp_path, recording(rec)).code == 0
    step = mw.CONFIG["chunk_rows"]
    blocks = [(r0, min(r0 + step, mw.H)) for r0 in range(0, mw.H, step)]
    assert len(blocks) == math.ceil(mw.H / step) and blocks[-1] == (18, 20)
    tr = exp["frames"]["train"]
    label = {f.id: f.label for f in tr}
    want = [
        (f.id, *rows)
        for k in exp["qualified"]
        for rows in blocks
        for f in tr
        if f.label == k
    ]
    assert sorted(rec.calls) == sorted(want)
    order = [(exp["qualified"].index(label[i]), r0) for i, r0, _ in rec.calls]
    assert order == sorted(order)  # classes of Q in order, each in row order
    parts = [f.id for p in ("train", "validation", "test") for f in exp["frames"][p]]
    assert [i for i in rec.whole if i in set(parts)] == parts
