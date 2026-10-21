"""Transcription of specs/acyclic.md for its tests, written from the spec alone: the exact
value of a YAML number, the verdict (1), the arcs and vertices of G (Notation), F by
graphs/feedback_oracle.py on the whole G, the labels (4), the shard files of (W) and the
`run.json` of (I) and (W)."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import yaml
from graphs.feedback_oracle import eades, removed_oracle

from .acyclic_world import json_bytes, name, sha

HALF = Fraction(1, 2)
KINDS = ("comparable", "removed", "incomparable", "dropped")


def exact(x) -> Fraction:
    """Notation: the exact value of a YAML number, the decimal as written."""
    return Fraction(str(x))


def verdict(a: dict, b: dict, labels: list, weights: dict, tau) -> tuple:
    """(1): (verdict, u or a, v or b, w or None); labels in request order, order 0 first."""
    s = len(labels) // 2
    valid = [(j, t) for j, t in enumerate(labels) if t in weights]
    refusals = [t for _, t in valid if weights[t] is None]
    if not valid:
        return "dropped", a, b, None
    if Fraction(len(refusals), len(valid)) > exact(tau):
        return "incomparable", a, b, None
    pis = [exact(weights[t]) if j < s else 1 - exact(weights[t])
           for j, t in valid if weights[t] is not None]  # fmt: skip
    p = sum(pis, Fraction(0)) / len(pis)
    return ("comparable", a, b, p) if p >= HALF else ("comparable", b, a, 1 - p)


@dataclass
class Graph:
    """G of Notation: n vertices, arcs and costs in shard order; arc[(k, j)] — the arc id
    of comparison j of shard k when comparable."""

    n: int
    arcs: list
    costs: list
    arc: dict


def graph(verdicts: list[list[tuple]]) -> Graph:
    """Arcs: the comparable verdicts in shard order; vertices: their clips (shard, id) in
    order of first appearance, u before v."""
    vertex, g = {}, Graph(0, [], [], {})
    for k, shard in enumerate(verdicts):
        for j, (kind, u, v, w) in enumerate(shard):
            if kind == "comparable":
                for x in (u, v):
                    vertex.setdefault((k, x["id"]), len(vertex))
                g.arc[(k, j)] = len(g.arcs)
                g.arcs.append((vertex[(k, u["id"])], vertex[(k, v["id"])]))
                g.costs.append(w - HALF)
    g.n = len(vertex)
    return g


def end(k: int, r: dict) -> dict:
    return {"shard": k, "id": r["id"], "episode": r["episode"]}


def shards(world: list, weights: dict, tau, f0: Callable = eades) -> list[tuple]:
    """(1)-(4) on the whole G: per shard (its pairs.json entries, its counts.json)."""
    verdicts = [[verdict(a, b, ls, weights, tau) for a, b, ls in s] for s in world]
    g = graph(verdicts)
    removed = set(removed_oracle(g.n, g.arcs, g.costs, f0))
    out = []
    for k, shard in enumerate(verdicts):
        pairs, counts = [], dict.fromkeys(KINDS, 0)
        for j, (kind, u, v, w) in enumerate(shard):
            if kind == "comparable":
                kind = "removed" if g.arc[(k, j)] in removed else kind
            counts[kind] += 1
            if kind != "dropped":
                p, inc = (float(w), 0) if kind == "comparable" else (None, 1)
                labels = {"p": p, "inc": inc}
                pairs.append({"a": end(k, u), "b": end(k, v), "labels": labels})
        out.append((pairs, counts))
    return out


def run_json(config: bytes, answers: list[bytes], pool: str, counts=None) -> bytes:
    """{out_dir}/run.json after (I) or, with the summed counts, after (W)."""
    value = {
        "config": yaml.safe_load(config),
        "sha256": {"config": sha(config), "votes": sha(b"".join(answers))},
        "pair_clips": pool,
    }
    if counts is not None:
        pairs = counts["comparable"] + counts["removed"] + counts["incomparable"]
        value |= {"complete": True, "counts": {"pairs": pairs, **counts}}
    return json_bytes(value)


def tree(expected: list[tuple], config: bytes, answers: list[bytes], pool: str) -> dict:
    """{out_dir} after a complete run: path relative to it -> bytes."""
    total = {kind: sum(c[kind] for _, c in expected) for kind in KINDS}
    files = {"run.json": run_json(config, answers, pool, total)}
    for k, (pairs, counts) in enumerate(expected):
        files[f"shards/{name(k)}/pairs.json"] = json_bytes(pairs)
        files[f"shards/{name(k)}/counts.json"] = json_bytes(counts)
    return files
