"""(1), (7), (8) of specs/policy.md: per clip, the input weights C, the actions whose
shift pi computes and the targets of P5."""

from collections.abc import Mapping

import numpy as np

from mcpref.datasets.clips.io import Clip
from mcpref.gui.mask import Masks, clip_masks

ALIGNMENT = "preceding-action-v1"


def frequencies(actions: list[str], index: Mapping[str, int], size: int) -> np.ndarray:
    """(1), (8): the count of each action over their number, `size` entries; 0 without
    actions."""
    ids = np.array([index[a] for a in actions], dtype=np.int64)
    counts = np.bincount(ids, minlength=size).astype(np.float64)
    return counts / len(actions) if actions else counts


def weights(clip: Clip, index: Mapping[str, int]) -> np.ndarray:
    """(1): preceding-interval frequencies at each state; state 0 has no action."""
    size = len(index) + 1
    none = np.zeros(size)
    none[-1] = 1.0
    rows = [
        frequencies(actions, index, size) if actions else none
        for actions in clip.info["actions"]
    ]
    return np.stack([none, *rows])


def next_actions(clip: Clip, index: Mapping[str, int]) -> np.ndarray:
    """(7): action of interval s when it is single, else 0; final state also 0."""
    info = clip.info
    out = np.zeros(len(info["states"]), dtype=np.int64)
    for s in range(len(info["states"]) - 1):
        if info["n"][s] == 1:
            out[s] = index[info["actions"][s][0]]
    return out


def targets(clip: Clip, masks: Masks) -> tuple[np.ndarray, np.ndarray]:
    """(8): interval s actions allowed at state s; shift valid for one allowed action."""
    index = {a: i for i, a in enumerate(masks.names)}
    M = clip_masks(masks, clip.info, clip.lines)
    actions, n = clip.info["actions"], clip.info["n"]
    q, chi = [], []
    for s in range(len(clip.info["states"]) - 1):
        T = [a for a in actions[s] if M[s, index[a]]]
        q.append(frequencies(T, index, len(index)))
        chi.append(n[s] == 1 and len(T) == 1)
    return np.stack(q), np.array(chi, dtype=bool)
