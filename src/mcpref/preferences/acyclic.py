"""Config, Behaviour and `main` of specs/acyclic.md: (R), (I), (W) with (3'), (4); (1) in
verdict.py, (2), (3) in graphs/feedback.py."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml

from mcpref import REPO, config_mapping, number, relative
from mcpref.cli.failure import FAILURES, failed
from mcpref.graphs.feedback import removed_arcs
from mcpref.preferences.verdict import HALF, Verdict, verdict
from mcpref.recognition.stored import shard_bar
from mcpref.runs import files
from mcpref.shards.store import complete_shards, shard_path, write_shard

PROG = "mcpref.preferences.acyclic"
COUNTS = ("comparable", "removed", "incomparable", "dropped")


@dataclass(frozen=True)
class AcyclicConfig:
    """Notation: configs/acyclic.yaml as written; paths relative to the repository root,
    resolved against it where read or written; exact values taken where used."""

    votes_dir: str
    out_dir: str
    weights: dict[str, int | float | None]
    refusal_share: int | float


def check_weights(weights: object) -> None:
    """Notation: a non-empty mapping from strings to numbers in [0, 1] or null, exactly
    one null."""
    if type(weights) is not dict or not weights:
        raise ValueError(f"weights {weights!r} is not a non-empty mapping")
    for t, w in weights.items():
        if type(t) is not str:
            raise ValueError(f"weights key {t!r} is not a string")
        if w is not None and not 0 <= number(f"weights.{t}", w) <= 1:
            raise ValueError(f"weights.{t} {w!r} is not in [0, 1]")
    if sum(w is None for w in weights.values()) != 1:
        raise ValueError(f"weights {weights!r} hold not exactly one null")


def load_config(path: Path) -> AcyclicConfig:
    """Interface: the YAML file, validated, into AcyclicConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, AcyclicConfig)
    for k in ("votes_dir", "out_dir"):
        relative(k, raw[k])
    check_weights(raw["weights"])
    tau = raw["refusal_share"]
    if not 0 <= number("refusal_share", tau) < 1:
        raise ValueError(f"refusal_share {tau!r} is not in [0, 1)")
    return AcyclicConfig(**raw)


def checks(cfg: AcyclicConfig, path: Path) -> dict:
    """(R): V_l complete; the keys of weights the tokens of its answers, in order."""
    run = files.finished_run(REPO / cfg.votes_dir, cfg.votes_dir, "complete")
    if list(cfg.weights) != [e["token"] for e in run["config"]["answers"]]:
        raise RuntimeError(
            f"weights of {path} differ from the answers of {cfg.votes_dir}"
        )
    return run


def header(cfg: AcyclicConfig, path: Path, votes_run: dict) -> dict:
    """(I): the fingerprint of the answers (bar `correction inputs`), the check of an
    existing {out_dir}/run.json, then run.json written."""
    K = complete_shards(REPO / cfg.votes_dir / "shards")
    with shard_bar("correction inputs", K, 0, K, 0, "shard") as update:
        votes = files.fingerprint(cfg.votes_dir, "answers.json", None, update)
    sha = {"config": files.sha256(path), "votes": votes}
    pair_clips = votes_run["sha256"]["pool"]
    out = REPO / cfg.out_dir / "run.json"
    files.check_fingerprints(out, sha | {"pair_clips": pair_clips}, cfg.out_dir)
    config = yaml.safe_load(path.read_text())
    head = {"config": config, "sha256": sha, "pair_clips": pair_clips}
    files.write_json(out, head)
    return head


def graph(verdicts: list[Verdict]) -> tuple[int, list, list]:
    """Notation: the arcs u -> v of a shard's comparable verdicts in order, costs w - 1/2;
    its vertices numbered by first appearance, u before v."""
    ids, arcs, costs = {}, [], []
    for kind, u, v, w in verdicts:
        if kind == "comparable":
            tail = ids.setdefault(u["id"], len(ids))
            arcs.append((tail, ids.setdefault(v["id"], len(ids))))
            costs.append(w - HALF)
    return len(ids), arcs, costs


def categories(verdicts: list[Verdict]) -> list[str]:
    """(3'), (4): per verdict, `comparable` (arc not in F), `removed` (arc in F),
    `incomparable` or `dropped`; F of the shard's graph alone."""
    F = set(removed_arcs(*graph(verdicts)))
    cats, e = [], 0
    for kind, *_ in verdicts:
        if kind == "comparable":
            kind = "removed" if e in F else kind
            e += 1
        cats.append(kind)
    return cats


def labels(category: str, w: Fraction | None) -> dict:
    """(4): the labels of a pair."""
    if category == "comparable":
        return {"p": float(w), "inc": 0}
    return {"p": None, "inc": 1}


def ref(k: int, x: dict) -> dict:
    """(W): a clip reference of pairs.json."""
    return {"shard": k, "id": x["id"], "episode": x["episode"]}


def shard_pairs(k: int, comparisons: list, cfg: AcyclicConfig) -> tuple[list, dict]:
    """(W): the pairs of shard k ((1), then (2), (3) by (3'), then (4), `dropped`
    omitted) and its counts."""
    verdicts = [
        verdict(c["a"], c["b"], [r["label"] for r in c["requests"]], cfg)
        for c in comparisons
    ]
    cats = categories(verdicts)
    pairs = [
        {"a": ref(k, x), "b": ref(k, y), "labels": labels(cat, w)}
        for cat, (_, x, y, w) in zip(cats, verdicts)
        if cat != "dropped"
    ]
    return pairs, {c: cats.count(c) for c in COUNTS}


def summed(out: Path, n: int) -> dict:
    """(W): the four counts summed over counts.json of shards 0..n-1, one at a time."""
    total = dict.fromkeys(COUNTS, 0)
    for k in range(n):
        counts = json.loads((shard_path(out, k) / "counts.json").read_text())
        total = {c: total[c] + counts[c] for c in COUNTS}
    return total


def write_shards(cfg: AcyclicConfig, votes_run: dict) -> None:
    """(W): shards h..K-1 of {out_dir}/shards/, each from its shard of V_l (bar
    `correction shards`, initial: the comparisons of the complete shards)."""
    votes, out = REPO / cfg.votes_dir / "shards", REPO / cfg.out_dir / "shards"
    K, h = complete_shards(votes), complete_shards(out)
    total, initial = votes_run["counts"]["comparisons"], sum(summed(out, h).values())
    with shard_bar("correction shards", K, h, total, initial, "comparison") as update:
        for k in range(h, K):
            comparisons = json.loads(
                (shard_path(votes, k) / "answers.json").read_text()
            )
            pairs, counts = shard_pairs(k, comparisons, cfg)
            with write_shard(out, k) as staged:
                (staged / "pairs.json").write_bytes(files.json_bytes(pairs))
                (staged / "counts.json").write_bytes(files.json_bytes(counts))
            update(len(comparisons))


def run(cfg: AcyclicConfig, path: Path) -> None:
    """Behaviour: (R), (I), (W); run.json rewritten complete, with the counts."""
    votes_run = checks(cfg, path)
    head = header(cfg, path, votes_run)
    write_shards(cfg, votes_run)
    out = REPO / cfg.out_dir
    c = summed(out / "shards", complete_shards(out / "shards"))
    counts = {"pairs": c["comparable"] + c["removed"] + c["incomparable"]} | c
    files.write_json(out / "run.json", head | {"complete": True, "counts": counts})


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.preferences.acyclic <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run(load_config(args.config), args.config)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
