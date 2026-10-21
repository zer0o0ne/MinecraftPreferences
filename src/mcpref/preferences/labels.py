"""Config, (R), (I), (L) and `main` of specs/vlm-labels.md; (1), (2) in prompt.py."""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from mcpref import REPO, config_mapping, integer, positive_int, relative
from mcpref.cli.failure import FAILURES, failed
from mcpref.preferences import prompt
from mcpref.recognition.stored import shard_bar
from mcpref.runs import files
from mcpref.shards.store import complete_shards, shard_path, write_shard
from mcpref.vlm.client import VLMConfig, ask_many
from mcpref.vlm.client import load_config as load_vlm_config

PROG = "mcpref.preferences.labels"
TEXTS = ("instruction", "first", "second", "question", "answer_line", "prefix")
TEXTS += ("inventory_line", "item_line", "no_items")
PLACEHOLDERS = {"question": ("{answers}",), "answer_line": ("{token}", "{description}")}
PLACEHOLDERS |= {"inventory_line": ("{items}",), "item_line": ("{type}", "{quantity}")}


@dataclass(frozen=True)
class VlmLabelsConfig:
    """Notation: configs/vlm-labels.yaml as written; paths relative to the root."""

    pool_dir: str
    vlm: str
    out_dir: str
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
    """Notation: the texts are strings holding their placeholders; prefix holds none of
    `*"' and no newline, which (2) removes or splits."""
    for k in TEXTS:
        marks = PLACEHOLDERS.get(k, ())
        if type(raw[k]) is not str or not all(m in raw[k] for m in marks):
            raise ValueError(f"{k} {raw[k]!r} is not a string holding {list(marks)}")
    if any(c in prompt.MARKS + "\n" for c in raw["prefix"]):
        raise ValueError(f"prefix {raw['prefix']!r} holds one of `*\"' or a newline")


def check_answers(answers: object) -> None:
    """Notation: A, a non-empty list of {token, description} strings; tokens distinct,
    non-empty, without whitespace or `*"'. and neither invalid nor no_answer."""
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
    """Interface: the YAML file, validated, into VlmLabelsConfig as written."""
    raw = config_mapping(yaml.safe_load(path.read_text()), path, VlmLabelsConfig)
    for k in ("pool_dir", "vlm", "out_dir"):
        relative(k, raw[k])
    check_texts(raw)
    check_answers(raw["answers"])
    for k in ("frames", "stride", "samples"):
        positive_int(k, raw[k])
    integer("seed", raw["seed"], 0, 2**32 - 1)
    return VlmLabelsConfig(**raw)


def checks(cfg: VlmLabelsConfig) -> dict:
    """(R): Q_c complete; (k_V - 1) s_V <= L - 1; Q_c's run.json."""
    run = files.finished_run(REPO / cfg.pool_dir, cfg.pool_dir, "clips")
    L = run["config"]["states"]
    if (cfg.frames - 1) * cfg.stride > L - 1:
        raise RuntimeError(f"frames and stride exceed the {L} states of {cfg.pool_dir}")
    return run


def inputs(cfg: VlmLabelsConfig, path: Path, run: dict, K: int) -> dict:
    """(I): Q_c's fingerprints (bar `label inputs`), checked; run.json written."""
    with shard_bar("label inputs", 2 * K, 0, 2 * K, 0, "shard") as update:
        fields = files.clip_fields(run)
        pool = files.fingerprint(cfg.pool_dir, "records.json", fields, update)
        comparisons = files.fingerprint(cfg.pool_dir, "comparisons.json", None, update)
    sha = {"config": files.sha256(path), "vlm": files.sha256(REPO / cfg.vlm)}
    sha |= {"pool": pool, "comparisons": comparisons}
    files.check_fingerprints(REPO / cfg.out_dir / "run.json", sha, cfg.out_dir)
    header = {"config": yaml.safe_load(path.read_text()), "sha256": sha}
    files.write_json(REPO / cfg.out_dir / "run.json", header)
    return header


def ask_shard(
    cfg: VlmLabelsConfig, vlm: VLMConfig, shard: Path, run: dict, start: int
) -> list[dict]:
    """(L): one ask_many call for the requests of the comparisons of a shard of Q_c
    (none without them), o = 0 showing (a, b), o = 1 (b, a); its answers.json."""
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


def ordered(tally: Counter, cfg: VlmLabelsConfig) -> dict:
    """(L): the labels that occur, tokens of A in order, then invalid, no_answer."""
    keys = [e["token"] for e in cfg.answers] + [prompt.INVALID, prompt.NO_ANSWER]
    return {k: tally[k] for k in keys if k in tally}


def summed(labels: Path, n: int) -> tuple[int, Counter, int]:
    """(L): the comparisons and labels of label shards 0..n-1 summed, counts.json read
    one at a time; `end` of shard n - 1 (0 when n = 0)."""
    comparisons, tally, end = 0, Counter(), 0
    for k in range(n):
        counts = json.loads((shard_path(labels, k) / "counts.json").read_bytes())
        comparisons, end = comparisons + counts["comparisons"], counts["end"]
        tally.update(counts["labels"])
    return comparisons, tally, end


def write_labels(
    labels: Path, k: int, entries: list, end: int, cfg: VlmLabelsConfig
) -> None:
    """Interface Datasets: label shard k, answers.json and counts.json, by write_shard."""
    tally = Counter(r["label"] for e in entries for r in e["requests"])
    counts = {"comparisons": len(entries), "labels": ordered(tally, cfg), "end": end}
    with write_shard(labels, k) as staged:
        (staged / "answers.json").write_bytes(files.json_bytes(entries))
        (staged / "counts.json").write_bytes(files.json_bytes(counts))


def label(cfg: VlmLabelsConfig, run: dict, header: dict, K: int) -> None:
    """(L): label shards h..K-1 (bar `label`), then run.json complete with `counts`."""
    pool, labels = REPO / cfg.pool_dir / "shards", REPO / cfg.out_dir / "shards"
    h = complete_shards(labels)
    initial, _, end = summed(labels, h)
    vlm, _ = load_vlm_config(REPO / cfg.vlm)
    with shard_bar("label", K, h, run["comparisons"], initial, "comparison") as update:
        for k in range(h, K):
            entries = ask_shard(cfg, vlm, shard_path(pool, k), run, end)
            end += 2 * cfg.samples * len(entries)
            write_labels(labels, k, entries, end, cfg)
            update(len(entries))
    comparisons, tally, _ = summed(labels, K)
    counts = {"comparisons": comparisons, "labels": ordered(tally, cfg)}
    done = header | {"complete": True, "counts": counts}
    files.write_json(REPO / cfg.out_dir / "run.json", done)


def main(argv: Sequence[str]) -> int:
    """`python -m mcpref.preferences.labels <config path>`; exit code."""
    parser = argparse.ArgumentParser(prog=f"python -m {PROG}")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        run = checks(cfg)
        K = complete_shards(REPO / cfg.pool_dir / "shards")
        label(cfg, run, inputs(cfg, args.config, run, K), K)
    except FAILURES as error:
        return failed(PROG, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
