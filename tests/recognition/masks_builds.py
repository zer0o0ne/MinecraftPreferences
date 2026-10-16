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
    """(13): built fold l -> (pairs of (b), (c), classes built in (2), its (b) results)."""
    out = {}
    for l in range(config["folds"]):
        if not any(fold[f.relpath] == l for f in tr):
            continue
        rest = [f for f in tr if fold[f.relpath] != l]
        sp_l = spread(rest, x, qualified, config)
        both = {k: v for k, v in sp_l.items() if k in sp}
        pairs = {"b": b_pairs(sp_l), "c": c_pairs(both, config) if len(sp) >= 2 else {}}
        built = [k for k in qualified if any(f.label == k for f in rest)]
        out[l] = (pairs, built, sp_l)
    return out


def catalogue_shards(pairs: dict) -> list:
    """templates/catalogue shard s, class s of K: <variant>/<member>/T and M of (a), (d)-(f)."""
    return [
        {
            f"{v}/{s[2:]}/{n}": a
            for v in "adef"
            for s, (t, m) in pairs[v].get(k, {}).items()
            for n, a in (("T", t), ("M", m))
        }
        for k in mw.CLASSES
    ]


def b_shards(qualified: list, sp: dict, built: dict) -> list:
    """templates/b shard s, class s of Q: <build>/T, sigma, M of its builds with a template."""
    builds = {"all": sp} | {f"fold{l}": b[2] for l, b in built.items()}
    return [
        {
            f"{name}/{n}": a
            for name, res in builds.items()
            if k in res
            for n, a in zip(("T", "sigma", "M"), res[k])
        }
        for k in qualified
    ]


def c_shards(pairs: dict, built: dict, folds: int) -> list:
    """templates/c shard s, build s (all, fold0, ...): class -> its (c) mask."""
    builds = [pairs["c"]] + [
        built[l][0]["c"] if l in built else {} for l in range(folds)
    ]
    return [{k: p[""][1] for k, p in c.items()} for c in builds]


def images(pairs: dict, qualified: list) -> dict:
    """PNG stem under templates/ -> (template image, mask) of the pairs built on all of F^tr:
    catalogue shards per member, (b) and (c) (shard 0, build all) per class; (f) grey
    floor(T / 1000)."""
    out = {}
    for s, k in enumerate(mw.CLASSES):
        for v in "adef":
            for m, (t, mask) in pairs[v].get(k, {}).items():
                image = t // 1000 if v == "f" else t
                out[f"catalogue/{s:06d}/{v}{m}"] = (image, mask)
    for s, k in enumerate(qualified):
        if k in pairs["b"]:
            out[f"b/{s:06d}/b__{k}"] = pairs["b"][k][""]
    for k, p in pairs["c"].items():
        out[f"c/000000/c__{k}"] = p[""]
    return out
