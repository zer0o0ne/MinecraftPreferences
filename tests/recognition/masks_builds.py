"""The pairs of specs/gui-masks.md (1)-(3), (10)-(13) on a world of masks_world.py, by the
naive references of masks_oracle.py and drawn_oracle.py."""

import functools

import drawn_oracle as d
import masks_oracle as o
import masks_world as mw

W, H = mw.W, mw.H


@functools.cache
def drawn() -> dict:
    """(1), (10)-(12): variant -> class -> name suffix -> (T, M), from the catalogue."""
    tex, phi, out = mw.textures(), mw.font(), {v: {} for v in "adef"}
    for k, cls in mw.CATALOGUE.items():
        outline = o.outline_pairs(cls, tex, W, H)
        built = {"a": {m: (d.rgb(c), mask) for m, (c, mask) in outline.items()}}
        built.update(
            d=d.label_pairs(cls, tex, phi, W, H), e=d.ring_pairs(cls, tex, W, H)
        )
        built["f"] = d.edge_pairs(cls, tex, W, H)
        for v, pairs in built.items():
            if pairs:
                out[v][k] = {f"__{m}": p for m, p in pairs.items()}
    return out


def qualified(tr: list, config: dict) -> list:
    """Q: the classes whose training frames span at least min_episodes episodes."""
    episodes = {k: {f.relpath for f in tr if f.label == k} for k in mw.CLASSES}
    return [k for k in mw.CLASSES if len(episodes[k]) >= config["min_episodes"]]


def spread(frames: list, x: dict, qualified: list, config: dict) -> dict:
    """(2) for each class of Q with a frame among `frames`: k -> (T, sigma, M), M not empty."""
    out = {}
    for k in qualified:
        fk = [x[f.id] for f in frames if f.label == k]
        if fk:
            t, s, m = o.spread_pair(fk, config["spread_quantile"], config["spread_max"])
            if m.any():
                out[k] = (t, s, m)
    return out


def b_pairs(sp: dict) -> dict:
    """(2): class -> {"": (T, M)} of the (b) results sp."""
    return {k: {"": (t, m)} for k, (t, _, m) in sp.items()}


def c_pairs(sp: dict, config: dict) -> dict:
    """(3) on the (b) results sp: class -> {"": (T, M)}; none for fewer than two classes."""
    if len(sp) < 2:
        return {}
    args = [{k: v[i] for k, v in sp.items()} for i in (0, 1)]
    n_p, eps = config["patches_per_class"], config["eps"]
    masks = o.patch_masks(*args, config["patch"], n_p, eps)
    return {k: {"": (sp[k][0], masks[k])} for k in masks}


def fold_pairs(
    tr: list, x: dict, qualified: list, sp: dict, fold: dict, config
) -> dict:
    """(13): built fold l -> (pairs of (b), (c), classes built in (2))."""
    out = {}
    for l in range(config["folds"]):
        if not any(fold[f.relpath] == l for f in tr):
            continue
        rest = [f for f in tr if fold[f.relpath] != l]
        sp_l = spread(rest, x, qualified, config)
        both = {k: v for k, v in sp_l.items() if k in sp}
        pairs = {"b": b_pairs(sp_l), "c": c_pairs(both, config) if len(sp) >= 2 else {}}
        out[l] = (pairs, [k for k in qualified if any(f.label == k for f in rest)])
    return out


def images(pairs: dict) -> dict:
    """templates/<variant>/<class>[__<member>]: (template image, mask); (f) grey T // 1000."""
    return {
        f"{v}/{k}{s}": (t // 1000 if v == "f" else t, m)
        for v in mw.VARIANTS
        for k, ps in pairs[v].items()
        for s, (t, m) in ps.items()
    }
