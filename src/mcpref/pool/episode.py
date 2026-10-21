"""(3) of specs/preference-pool.md: an episode of a draw, checked and processed as a
candidate of perception-data."""

from functools import partial
from typing import TYPE_CHECKING

import httpx
import numpy as np

from mcpref.clips.episode import Candidate, Pipeline, process
from mcpref.clips.plan import STORED, checked
from mcpref.clips.store import rejected, staging
from mcpref.vpt.downloads import fetch

if TYPE_CHECKING:
    from mcpref.pool.run import PreferencePoolConfig

Work = tuple[dict, list[str], bytes | None]
"""(3): an episode's record, the ids of its staged clips, its options file to store."""


def episode(
    pipe: Pipeline,
    cfg: "PreferencePoolConfig",
    k: int,
    j: int,
    subset: str,
    basedir: str,
    relpath: str,
) -> Work:
    """(3): episode j of draw k on a client of its own: the checks of perception-data, a
    candidate by its (2)-(5) with m_p, PCG64([seed, 4, k, j]), staging/<k>/<j>.mp4."""
    p = pipe.cfg
    with httpx.Client(headers={"x-ms-version": p.x_ms_version}) as client:
        ex = checked(p, partial(fetch, client, p), basedir, relpath)
        options = ex.options_file if ex.reason in STORED else None
        if ex.reason is not None:
            return rejected(subset, relpath, ex), [], options
        stem = basedir + relpath.removesuffix(".mp4")
        video = staging(pipe.out, k) / f"{j}.mp4"
        g = np.random.Generator(np.random.PCG64([cfg.seed, 4, k, j]))
        cand = Candidate(k, subset, relpath, stem, ex, video, g, cfg.per_episode)
        record, ids = process(pipe, cand, client)
    return record, ids, options
