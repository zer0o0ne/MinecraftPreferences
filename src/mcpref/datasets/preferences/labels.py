"""Config, (R), (I), (L) and `main` of specs/vlm-labels.md; (1), (2) in prompt.py."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcpref.config import (
    REPO,
    config_mapping,
    integer,
    positive_int,
    read_yaml,
    relative,
)
from mcpref.datasets.preferences import prompt
from mcpref.datasets.preferences.acyclic import load_config as load_acyclic_config
from mcpref.datasets.preferences.labels_inputs import checks, inputs
from mcpref.datasets.preferences.producer import ordered, summed, write_vote
from mcpref.diagnostics.core.runtime import DiagnosticRun, diagnostic_bar
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.progress import shard_bar
from mcpref.infrastructure.runs import files
from mcpref.infrastructure.shards.store import complete_shards, shard_path
from mcpref.integrations.vlm.client import VLMConfig, ask_many
from mcpref.integrations.vlm.client import load_config as load_vlm_config

PROG = "mcpref.datasets.preferences.labels"
TEXTS = ("instruction", "first", "second", "question", "answer_line", "prefix")
TEXTS += ("inventory_line", "item_line", "no_items")
PLACEHOLDERS = {"question": ("{answers}",), "answer_line": ("{token}", "{description}")}
PLACEHOLDERS |= {"inventory_line": ("{items}",), "item_line": ("{type}", "{quantity}")}


@dataclass(frozen=True)
class VlmLabelsConfig:
    """Notation: configs/datasets/preferences/labels.yaml as written; paths relative to the root."""

    pool_dir: str
    vlm: str
    out_dir: str
    diagnostics: str
    acyclic: str
    instruction: str
    first: str
    second: str
    question: str
    answer_line: str
    prefix: str
    inventory_line: str
    item_line: str
    no_items: str
    answers: list[dict]
    frames: int
    stride: int
    samples: int
    seed: int


def check_texts(raw: dict) -> None:
    for k in TEXTS:
        marks = PLACEHOLDERS.get(k, ())
        if type(raw[k]) is not str or not all(m in raw[k] for m in marks):
            raise ValueError(f"{k} {raw[k]!r} is not a string holding {list(marks)}")
    if any(c in prompt.MARKS + "\n" for c in raw["prefix"]):
        raise ValueError(f"prefix {raw['prefix']!r} holds one of `*\"' or a newline")


def check_answers(answers: object) -> None:
    keys = {"token", "description"}
    if not (type(answers) is list and answers) or not all(
        type(e) is dict and set(e) == keys and all(type(e[k]) is str for k in keys)
        for e in answers
    ):
        raise ValueError(f"answers {answers!r} is not a list of {{token, description}}")
    tokens = [e["token"] for e in answers]
    for t in tokens:
        bad = not t or t in (prompt.INVALID, prompt.NO_ANSWER) or tokens.count(t) > 1
        if bad or any(c.isspace() or c in prompt.MARKS + "." for c in t):
            raise ValueError(f"token {t!r} is empty, reserved, repeated or not plain")


def load_config(path: Path) -> VlmLabelsConfig:
    raw = config_mapping(read_yaml(path), path, VlmLabelsConfig)
    for k in ("pool_dir", "vlm", "out_dir", "diagnostics", "acyclic"):
        relative(k, raw[k])
    check_texts(raw)
    check_answers(raw["answers"])
    for k in ("frames", "stride", "samples"):
        positive_int(k, raw[k])
    integer("seed", raw["seed"], 0, 2**32 - 1)
    return VlmLabelsConfig(**raw)


def ask_shard(
    cfg: VlmLabelsConfig, vlm: VLMConfig, shard: Path, run: dict, start: int
) -> list[dict]:
    comparisons = json.loads((shard / "comparisons.json").read_bytes())
    ids = dict.fromkeys(x["id"] for ab in comparisons for x in (ab["a"], ab["b"]))
    shown = {i: prompt.read_shown(shard / "clips" / i, cfg, run) for i in ids}
    shows = [(shown[ab["a"]["id"]], shown[ab["b"]["id"]]) for ab in comparisons]
    reqs = prompt.requests(cfg, len(comparisons), start)
    prompts = [
        prompt.prompt(cfg, *(shows[c] if o == 0 else shows[c][::-1]))
        for c, o, _, _ in reqs
    ]
    answers = ask_many(prompts, [r[3] for r in reqs], vlm) if reqs else []
    entries = [{"a": ab["a"], "b": ab["b"], "requests": []} for ab in comparisons]
    for (c, o, i, seed), answer in zip(reqs, answers):
        request = {"order": o, "sample": i, "seed": seed, "answer": answer}
        entries[c]["requests"].append(request | {"label": prompt.parse(answer, cfg)})
    return entries


def write_labels(
    labels: Path,
    k: int,
    entries: list,
    end: int,
    cfg: VlmLabelsConfig,
    *,
    diagnostics: DiagnosticRun,
    weights: Mapping[str, float | None],
) -> None:
    """Persist vote observation before its source shard, then accept that exact shard."""
    write_vote(labels, k, entries, end, cfg, diagnostics=diagnostics, weights=weights)


def label(
    cfg: VlmLabelsConfig,
    run: dict,
    header: dict,
    K: int,
    *,
    diagnostics: DiagnosticRun,
    weights: Mapping[str, float | None],
) -> None:
    pool, labels = REPO / cfg.pool_dir / "shards", REPO / cfg.out_dir / "shards"
    h = complete_shards(labels)
    watermark = diagnostics.state["channels"].get("vote", {}).get("watermark", h)
    diagnostics.register("vote", watermark=watermark, next=watermark)
    diagnostics.recover("vote")
    h = complete_shards(labels)
    initial, _, end = summed(labels, h)
    vlm, _ = load_vlm_config(REPO / cfg.vlm)
    accepted = diagnostics.state["channels"]["vote"]["next"] - watermark
    with (
        shard_bar("label", K, h, run["comparisons"], initial, "comparison") as update,
        diagnostic_bar(
            "diagnostics targets", K - watermark, accepted, "shard"
        ) as diagnostic,
    ):
        for k in range(h, K):
            entries = ask_shard(cfg, vlm, shard_path(pool, k), run, end)
            end += 2 * cfg.samples * len(entries)
            write_labels(
                labels, k, entries, end, cfg, diagnostics=diagnostics, weights=weights
            )
            update(len(entries))
            if diagnostic is not None:
                diagnostic.update()
    comparisons, tally, _ = summed(labels, K)
    counts = {"comparisons": comparisons, "labels": ordered(tally, cfg)}
    done = header | {"complete": True, "counts": counts}
    files.write_json(REPO / cfg.out_dir / "run.json", done)
    diagnostics.finish("vote", source_complete=True, done=complete_shards(labels) == K)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        diagnostics = DiagnosticRun.load(REPO / cfg.diagnostics)
        weights = load_acyclic_config(REPO / cfg.acyclic).weights
        run = checks(cfg)
        K = complete_shards(REPO / cfg.pool_dir / "shards")
        label(
            cfg,
            run,
            inputs(cfg, args.config, run, K),
            K,
            diagnostics=diagnostics,
            weights=weights,
        )
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
