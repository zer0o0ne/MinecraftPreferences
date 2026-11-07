"""Config, Behaviour and `main` of specs/acyclic.md, including stage-33a handoff."""

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
from mcpref.diagnostics.runtime import DiagnosticRun, diagnostic_bar
from mcpref.graphs.feedback import removed_arcs
from mcpref.preferences.producer import accept_acyclic, stage_acyclic
from mcpref.preferences.verdict import HALF, Verdict, verdict
from mcpref.recognition.stored import shard_bar
from mcpref.runs import files
from mcpref.shards.store import complete_shards, shard_path, write_shard

PROG = "mcpref.preferences.acyclic"
COUNTS = ("comparable", "removed", "incomparable", "dropped")


@dataclass(frozen=True)
class AcyclicConfig:
    votes_dir: str
    out_dir: str
    diagnostics: str
    weights: dict[str, int | float | None]
    refusal_share: int | float


def check_weights(weights: object) -> None:
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
    raw = config_mapping(yaml.safe_load(path.read_text()), path, AcyclicConfig)
    for k in ("votes_dir", "out_dir", "diagnostics"):
        relative(k, raw[k])
    check_weights(raw["weights"])
    tau = raw["refusal_share"]
    if not 0 <= number("refusal_share", tau) < 1:
        raise ValueError(f"refusal_share {tau!r} is not in [0, 1)")
    return AcyclicConfig(**raw)


def checks(cfg: AcyclicConfig, path: Path) -> dict:
    run = files.finished_run(REPO / cfg.votes_dir, cfg.votes_dir, "complete")
    if list(cfg.weights) != [e["token"] for e in run["config"]["answers"]]:
        raise RuntimeError(
            f"weights of {path} differ from the answers of {cfg.votes_dir}"
        )
    return run


def header(cfg: AcyclicConfig, path: Path, votes_run: dict) -> dict:
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
    ids, arcs, costs = {}, [], []
    for kind, u, v, w in verdicts:
        if kind == "comparable":
            tail = ids.setdefault(u["id"], len(ids))
            arcs.append((tail, ids.setdefault(v["id"], len(ids))))
            costs.append(w - HALF)
    return len(ids), arcs, costs


def categories(verdicts: list[Verdict]) -> list[str]:
    F = set(removed_arcs(*graph(verdicts)))
    cats, e = [], 0
    for kind, *_ in verdicts:
        if kind == "comparable":
            kind = "removed" if e in F else kind
            e += 1
        cats.append(kind)
    return cats


def labels(category: str, w: Fraction | None) -> dict:
    if category == "comparable":
        return {"p": float(w), "inc": 0}
    return {"p": None, "inc": 1}


def ref(k: int, x: dict) -> dict:
    return {"shard": k, "id": x["id"], "episode": x["episode"]}


def shard_pairs(
    k: int, comparisons: list, cfg: AcyclicConfig
) -> tuple[list, dict, list[str]]:
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
    return pairs, {c: cats.count(c) for c in COUNTS}, cats


def summed(out: Path, n: int) -> dict:
    total = dict.fromkeys(COUNTS, 0)
    for k in range(n):
        counts = json.loads((shard_path(out, k) / "counts.json").read_text())
        total = {c: total[c] + counts[c] for c in COUNTS}
    return total


def write_shards(
    cfg: AcyclicConfig, votes_run: dict, diagnostics: DiagnosticRun
) -> None:
    votes, out = REPO / cfg.votes_dir / "shards", REPO / cfg.out_dir / "shards"
    K, h = complete_shards(votes), complete_shards(out)
    channels = diagnostics.state["channels"]
    label_watermark = channels.get("label", {}).get("watermark", h)
    graph_watermark = channels.get("graph", {}).get("watermark", h)
    diagnostics.register("label", watermark=label_watermark, next=label_watermark)
    diagnostics.register("graph", watermark=graph_watermark, next=graph_watermark)
    diagnostics.recover("label")
    diagnostics.recover("graph")
    h = complete_shards(out)
    total, initial = votes_run["counts"]["comparisons"], sum(summed(out, h).values())
    diagnostic_initial = channels["graph"]["next"] - graph_watermark
    with (
        shard_bar("correction shards", K, h, total, initial, "comparison") as update,
        diagnostic_bar(
            "diagnostics targets", K - graph_watermark, diagnostic_initial, "shard"
        ) as diagnostic,
    ):
        for k in range(h, K):
            comparisons = json.loads(
                (shard_path(votes, k) / "answers.json").read_text()
            )
            pairs, counts, cats = shard_pairs(k, comparisons, cfg)
            stage_acyclic(k, comparisons, pairs, cats, cfg, diagnostics)
            with write_shard(out, k) as staged:
                (staged / "pairs.json").write_bytes(files.json_bytes(pairs))
                (staged / "counts.json").write_bytes(files.json_bytes(counts))
            accept_acyclic(diagnostics, k)
            update(len(comparisons))
            if diagnostic is not None:
                diagnostic.update()


def run(cfg: AcyclicConfig, path: Path) -> None:
    diagnostics = DiagnosticRun.load(REPO / cfg.diagnostics)
    votes_run = checks(cfg, path)
    head = header(cfg, path, votes_run)
    write_shards(cfg, votes_run, diagnostics)
    out = REPO / cfg.out_dir
    c = summed(out / "shards", complete_shards(out / "shards"))
    counts = {"pairs": c["comparable"] + c["removed"] + c["incomparable"]} | c
    files.write_json(out / "run.json", head | {"complete": True, "counts": counts})
    done = diagnostics.state["channels"]["label"]["next"] == complete_shards(
        REPO / cfg.votes_dir / "shards"
    )
    diagnostics.finish("label", source_complete=True, done=done)
    diagnostics.finish("graph", source_complete=True, done=done)


def main(argv: Sequence[str]) -> int:
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
