"""(1), (7), (8) of specs/policy.md: per clip, the input weights C, the actions whose
shift pi computes and the targets of P5."""

from collections.abc import Mapping

import numpy as np

from mcpref.gui.mask import Masks, clip_masks
from mcpref.pretrain.clip import Clip


def frequencies(actions: list[str], index: Mapping[str, int], size: int) -> np.ndarray:
    """(1), (8): the count of each action over their number, `size` entries; 0 without
    actions."""
    ids = np.array([index[a] for a in actions], dtype=np.int64)
    counts = np.bincount(ids, minlength=size).astype(np.float64)
    return counts / len(actions) if actions else counts


def weights(clip: Clip, index: Mapping[str, int]) -> np.ndarray:
    """(1): C of the clip, shape (L, |A| + 1): the frequencies of interval j's actions,
    the last row the one-hot of no action."""
    size = len(index) + 1
    rows = [frequencies(actions, index, size) for actions in clip.info["actions"]]
    none = np.zeros(size)
    none[-1] = 1.0
    return np.stack([*rows, none])


def next_actions(clip: Clip, index: Mapping[str, int]) -> np.ndarray:
    """(7): a_s = the action of interval s + 1 when s <= L - 3 and n_{s+1} = 1, else
    the placeholder 0; shape (L,), int64."""
    info = clip.info
    out = np.zeros(len(info["states"]), dtype=np.int64)
    for s in range(len(info["states"]) - 2):
        if info["n"][s + 1] == 1:
            out[s] = index[info["actions"][s + 1][0]]
    return out


def targets(clip: Clip, masks: Masks) -> tuple[np.ndarray, np.ndarray]:
    """(8): q_s of shape (L - 2, |A|), from the actions of interval s + 1 allowed at
    state s + 1; chi_s = [n_s = 1][n_{s+1} = 1][n'_s = 1], shape (L - 2,)."""
    index = {a: i for i, a in enumerate(masks.names)}
    M = clip_masks(masks, clip.info, clip.lines)
    actions, n = clip.info["actions"], clip.info["n"]
    q, chi = [], []
    for s in range(len(clip.info["states"]) - 2):
        T = [a for a in actions[s + 1] if M[s + 1, index[a]]]
        q.append(frequencies(T, index, len(index)))
        chi.append(n[s] == 1 and n[s + 1] == 1 and len(T) == 1)
    return np.stack(q), np.array(chi, dtype=bool)
