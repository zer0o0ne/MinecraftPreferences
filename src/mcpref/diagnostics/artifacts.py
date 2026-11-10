"""Atomic accepted diagnostic records, typed pairs and streaming record index."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path

from mcpref import REPO
from mcpref.diagnostics.types import MetricRecord, ReferencePair
from mcpref.runs.files import json_bytes
from mcpref.shards.store import (
    complete_shards,
    make_dirs,
    name,
    part_path,
    publish,
    shard_path,
    write_file,
    write_shard,
)

TARGET_PLOTS = (
    ("raw-vote-0", ["target.vote_lt_0_2.order0", "target.vote_gt_0_8.order0"]),
    ("raw-vote-1", ["target.vote_lt_0_2.order1", "target.vote_gt_0_8.order1"]),
    ("graph-removed", ["target.graph_removed"]),
    (
        "all-pairs",
        ["target.graph_removed_all_pairs", "target.semantic_inc", "target.all_inc"],
    ),
    ("aggregate", ["target.aggregated_gt_0_8"]),
    ("model", ["target.model_lt_0_2", "target.model_gt_0_8"]),
)
PRETRAIN_PANELS = {
    "losses": ["P", "P_inc", "P_succ"],
    "progress": ["pretrain.progress"],
    "brier-inc": ["preference.inc.brier"],
    "brier-succ": ["preference.succ.brier"],
    "calibration-inc": ["preference.inc.ece"],
    "calibration-succ": ["preference.succ.ece"],
}
PLOT_REGISTRY = tuple(
    {
        "group": "target-provenance",
        "file": f"target-provenance__{facet}.svg",
        "series": series,
        "x": "index",
    }
    for facet, series in TARGET_PLOTS
)


def pretrain_registry(facets: Sequence[tuple[str, str]]) -> list[dict]:
    return [
        {
            "group": "pretrain-preference",
            "file": f"pretrain-preference__{split}__{condition}__{panel}.svg",
            "series": series,
            "x": "step",
        }
        for split, condition in facets
        for panel, series in PRETRAIN_PANELS.items()
    ]


def write_record(out_dir: Path, kind: str, index: int, record: MetricRecord) -> None:
    if record["kind"] != kind or index != complete_records(out_dir, kind):
        raise ValueError("diagnostic record indices must be contiguous by kind")
    accepted = dict(record) | {"index": index}
    with write_shard(out_dir / kind, index) as staged:
        (staged / "metrics.json").write_bytes(json_bytes(accepted))


def write_pairs(path: Path, pairs: Sequence[ReferencePair]) -> None:
    write_file(path / "pairs.json", json_bytes([asdict(pair) for pair in pairs]))


def read_pairs(path: Path) -> list[ReferencePair]:
    return [
        ReferencePair(**pair) for pair in json.loads((path / "pairs.json").read_bytes())
    ]


def complete_records(out_dir: Path, kind: str) -> int:
    return complete_shards(out_dir / kind)


def iter_records(out_dir: Path, kind: str) -> Iterator[MetricRecord]:
    for index in range(complete_records(out_dir, kind)):
        path = shard_path(out_dir / kind, index) / "metrics.json"
        yield json.loads(path.read_bytes())


def iter_all_records(out_dir: Path) -> Iterator[MetricRecord]:
    for kind in ("targets", "pretrain", "rl", "mcts", "milestones"):
        yield from iter_records(out_dir, kind)


def input_identity(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO / path).read_bytes())
    return digest.hexdigest()


def index_entry(record: MetricRecord) -> dict:
    return {
        "kind": record["kind"],
        "index": record["index"],
        "id": record["id"],
        "path": f"{record['kind']}/{name(record['index'])}/metrics.json",
        "condition": record["context"]["condition"],
        "clusters": record["populations"],
    }


def stream_index(
    out_dir: Path, records: Iterable[MetricRecord], plots: Sequence[dict]
) -> None:
    path = out_dir / "index.json"
    make_dirs(path.parent)
    part = part_path(path)
    with part.open("w", encoding="utf-8") as stream:
        stream.write('{\n  "version": 1,\n  "records": [')
        count = 0
        for record in records:
            stream.write(",\n" if count else "\n")
            entry = json.dumps(index_entry(record), indent=2)
            stream.write("\n".join(f"    {line}" for line in entry.splitlines()))
            count += 1
        stream.write("\n  " if count else "")
        stream.write('],\n  "plots": ')
        rendered = json.dumps(list(plots), indent=2)
        stream.write(rendered.replace("\n", "\n  "))
        stream.write("\n}\n")
    publish(part, path)


def previous_plots(out_dir: Path) -> list[dict]:
    path = out_dir / "index.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith('  "plots":'):
                payload = line.split(":", 1)[1].strip() + stream.read()
                return json.JSONDecoder().raw_decode(payload)[0]
    return []


def write_accepted_index(out_dir: Path) -> None:
    stream_index(out_dir, iter_all_records(out_dir), previous_plots(out_dir))


def write_index(out_dir: Path, records: Sequence[MetricRecord]) -> None:
    stream_index(out_dir, records, previous_plots(out_dir))


GROUP_PANELS = {
    "policy-action-quality": {
        "ce": [
            "policy.nll",
            "policy.uniform_allowed_ce",
            "policy.prior_ce",
            "policy.copy_ce",
        ],
        "distributions": [
            "policy.brier",
            "policy.tv",
            "policy.entropy",
            "policy.teacher_entropy",
        ],
        "support": [
            "policy.top1_support",
            "policy.topk_support",
            "policy.topk_target_mass",
        ],
        "confusion": [
            "policy.micro_recall",
            "policy.macro_recall",
            "policy.macro_precision",
        ],
    },
    "policy-calibration": {"calibration": ["policy.confidence", "policy.ece"]},
    "policy-masks": {
        "count": ["mask.count"],
        "rates": ["mask.empty", "mask.singleton", "mask.executed_violation"],
        "probability": ["mask.probability", "mask.allowed_entropy"],
        "teacher": ["policy.singleton_teacher", "policy.zero_teacher"],
    },
    "ppo": {
        "ratio": ["ppo.ratio", "ppo.ratio_max", "ppo.logprob_drift", "ppo.action_tv"],
        "advantage": ["ppo.advantage", "ppo.advantage_std"],
        "kl": ["ppo.kl_sample", "ppo.kl_exact"],
        "ess": ["ppo.ess", "ppo.ess_normalized"],
        "clip": ["ppo.clip_positive", "ppo.clip_negative"],
        "counts": ["ppo.eligible", "ppo.fresh", "ppo.pending"],
    },
    "policy-mcts": {
        "agreement": ["mcts.top1_prior_agreement", "mcts.top1_policy_agreement"],
        "ranks": ["mcts.rank_prior", "mcts.rank_visits"],
        "divergence": ["mcts.kl_prior_visits", "mcts.js_prior_visits"],
        "exploration": [
            "mcts.policy_support_mass",
            "mcts.visit_entropy",
            "mcts.top_width_prior_mass",
        ],
    },
    "mcts-outcomes": {
        "hits": [
            "mcts.chosen_hit",
            "mcts.top1_hit",
            "mcts.topk_hit",
            "mcts.support_hit",
        ],
        "matched-mass": ["mcts.matched_mass", "mcts.matched_hit"],
        "errors": ["mcts.d_alpha_error", "mcts.zero_shift_error"],
    },
    "factual-metra": {
        "shift": ["shift.error", "shift.error_squared", "shift.zero_baseline"],
        "metra": ["metra.d", "metra.q", "metra.lambda"],
        "rates": ["metra.constraint_active", "metra.zero_skill"],
    },
    "update-health": {
        "global": [
            "update.preclip_norm",
            "update.clip_coefficient",
            "update.learning_rate",
        ],
        "head": [
            "update.head_row_norm",
            "update.head_update_row_norm",
            "update.logit_norm",
        ],
        "rows": [
            "update.zero_gradient_row",
            "update.unsupported_action",
            "update.zero_unsupported_row",
        ],
    },
    "memory-efficiency": {
        "counts": [
            "memory.seeds_per_real_state",
            "memory.temporal_per_real_state",
            "memory.nearest_per_real_state",
            "memory.beam_per_real_state",
            "memory.evictions",
            "memory.unused",
        ],
        "ratios": [
            "memory.dedup_ratio",
            "memory.repeated_comparison",
            "memory.capacity",
        ],
        "coverage": [
            "memory.episode_coverage",
            "memory.empty_candidates_per_real_state",
        ],
    },
    "losses": {
        "total": ["loss", "phi", "pi", "P"],
        "components": ["ppo", "shift", "metra", "P_inc", "P_succ"],
    },
}
for _milestone in (
    "stone_pickaxe",
    "iron",
    "iron_pickaxe",
    "diamonds",
    "obsidian",
    "blaze_rod",
):
    GROUP_PANELS.setdefault("milestones", {})[f"coverage-{_milestone}"] = [
        f"milestones.coverage_{_milestone}"
    ]
    GROUP_PANELS.setdefault("milestones", {})[f"time-{_milestone}"] = [
        f"milestones.time_to_first_{_milestone}"
    ]
GROUP_PANELS["milestones"]["first-times"] = [
    f"joined.first_times.{name}"
    for name in (
        "stone_pickaxe",
        "iron",
        "iron_pickaxe",
        "diamonds",
        "obsidian",
        "blaze_rod",
    )
]


def group_registry(
    group: str, facets: Sequence[tuple[str, str]], extra: Sequence[str] = ()
) -> list[dict]:
    entries = [
        {
            "group": group,
            "file": f"{group}__{split}__{condition}__{panel}.svg",
            "series": series,
            "x": "step",
        }
        for split, condition in facets
        for panel, series in GROUP_PANELS[group].items()
    ]
    for split, condition in facets:
        prefix = f"{group}__{split}__{condition}"
        if extra:
            entries.append(
                {
                    "group": group,
                    "file": f"{prefix}__groups.svg",
                    "series": list(extra),
                    "x": "step",
                }
            )
        if group == "policy-calibration":
            entries.extend(
                [
                    {
                        "group": group,
                        "file": f"{prefix}__reliability.svg",
                        "series": ["confidence", "accuracy"],
                        "x": "bin",
                    },
                    {
                        "group": group,
                        "file": f"{prefix}__support.svg",
                        "series": ["n"],
                        "x": "bin",
                    },
                ]
            )
        if group == "mcts-outcomes":
            entries.append(
                {
                    "group": group,
                    "file": f"{prefix}__matched-reliability.svg",
                    "series": ["hit"],
                    "x": "matched_mass",
                }
            )
        if group == "milestones":
            for name in (
                "stone_pickaxe",
                "iron",
                "iron_pickaxe",
                "diamonds",
                "obsidian",
                "blaze_rod",
            ):
                entries.append(
                    {
                        "group": group,
                        "file": f"{prefix}__km-{name}.svg",
                        "series": ["survival"],
                        "x": f"km:{name}",
                    }
                )
    return entries
