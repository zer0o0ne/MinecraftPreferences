"""Locate preference-bearing updates in the shuffled observation schedule."""

import numpy as np
from tqdm import tqdm

from mcpref.config import REPO
from mcpref.gui.mask import load_config, load_masks
from mcpref.training.pretrain.continuous.data.metadata import rows


def gradient_steps(context):
    if (
        not context.cfg.gradient_diagnostics
        or not context.index.run["populations"]["train"]["N_P"]
    ):
        return {}
    index, cfg = context.index, context.cfg
    root = REPO / index.run["config"]["dataset"]
    masks = load_masks(load_config(REPO / cfg.action_mask))
    fragments = list(index.fragments("train"))
    ends = {}
    for fragment in tqdm(
        fragments,
        total=len(fragments),
        desc="Continuous gradient selection",
        unit="fragment",
    ):
        needed = {}
        for key, completion in index.db.execute(
            "SELECT id,completion FROM endpoints WHERE fragment=? AND degree>0",
            (fragment["id"],),
        ):
            needed.setdefault(completion, []).append(key)
        if not needed:
            continue
        for row in rows(root, fragment, masks):
            for key in needed.get(row["id"], ()):
                ends[key] = (fragment["id"], row["index"])
    generator = np.random.Generator(np.random.PCG64(cfg.seed))
    counts = {fragment["id"]: fragment["observations"] for fragment in fragments}
    candidates, offset = set(), 0
    for _ in range(cfg.epochs):
        starts, count = {}, 0
        for key in generator.permutation(list(counts)):
            starts[key] = count
            count += counts[key]
        for a, b in index.db.execute(
            "SELECT a,b FROM edges WHERE part='train' ORDER BY id"
        ):
            candidates.add(
                offset
                + max(
                    (starts[ends[key][0]] + ends[key][1]) // cfg.batch_size + 1
                    for key in (a, b)
                )
            )
        offset += (count + cfg.batch_size - 1) // cfg.batch_size
    return select_roles(candidates, context.total)


def select_roles(candidates, total):
    if not candidates:
        return {}
    steps = sorted(candidates)
    middle = next((step for step in steps if step >= max(1, total // 2)), steps[-1])
    result = {}
    for role, step in zip(
        ("beginning", "middle", "end"), (steps[0], middle, steps[-1])
    ):
        result.setdefault(step, []).append(role)
    return result
