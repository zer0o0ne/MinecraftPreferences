"""Collect and migrate continuous streams; specs/continuous-data.md."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from mcpref.config import REPO
from mcpref.datasets.clips.collect import load_config as load_source_config
from mcpref.datasets.clips.plan import pipeline
from mcpref.datasets.clips.run import gamma_of
from mcpref.datasets.continuous import inputs, mapping
from mcpref.datasets.continuous.boundaries import blocks
from mcpref.datasets.continuous.config import load_config, plans
from mcpref.datasets.continuous.frames import verify
from mcpref.datasets.continuous.recovery import cleanup, recover, seal
from mcpref.datasets.continuous.rows import Rows
from mcpref.datasets.continuous.sources import acquire
from mcpref.datasets.continuous.store import jsonl, write_timeline
from mcpref.datasets.continuous.timeline import compile_timeline, identity
from mcpref.datasets.continuous.views.clips import clip_info
from mcpref.datasets.vpt.episodes import options_ok
from mcpref.datasets.vpt.video.sequence import RowView
from mcpref.infrastructure.cli.failure import FAILURES, failed
from mcpref.infrastructure.runs.files import json_bytes, write_json
from mcpref.infrastructure.shards.store import complete_shards, shard_path, write_shard
from mcpref.training.pretrain.classical.data.index import (
    load_config as load_split_config,
)


def prepare(pipe, cfg, pieces, rows: Rows, fingerprint: dict):
    sources = []
    for piece in tqdm(
        pieces,
        total=len(pieces),
        desc="Fragment sources",
        unit="piece",
        leave=False,
        mininterval=10,
    ):
        target, receipt = acquire(pipe, cfg, piece)
        source = {
            "piece": piece,
            "target": target,
            "reason": receipt.get("reason"),
            "offset": None,
            "rows": piece["nominal_steps"],
        }
        if source["reason"] is None:
            options = json.loads((target / "options.json").read_text())
            if not options_ok(options, pipe.cfg.required_options):
                source["reason"] = "options"
            else:
                try:
                    source["offset"] = rows.append_source(
                        target / "source.jsonl.gz", piece
                    )
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    source |= {"reason": "malformed_jsonl", "detail": str(error)}
        if source["reason"] is None:
            view = RowView(
                rows, range(source["offset"], source["offset"] + source["rows"])
            )
            source["validation"] = verify(
                pipe, target, view, fingerprint | {"source_files": receipt["files"]}
            )
            source["reason"] = source["validation"]["reason"]
        sources.append(source)
    return sources


def process_fragment(
    pipe, cfg, out, k, fragment, pieces, by_source, worlds, first_shard, fingerprint
):
    temporary = out / "staging" / "rows.sqlite"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    rows = Rows(temporary, cfg.metadata_rows)
    streams, mapped, clips = [], [], {}
    try:
        sources = prepare(pipe, cfg, pieces, rows, fingerprint)
        index = first_shard
        for group in blocks(sources, rows, cfg.join_gap_ms):
            if group[0]["reason"] is not None:
                mapped.extend(
                    mapping.unavailable(c, s["reason"])
                    for s in group
                    for c in by_source.get(s["piece"]["relpath"], [])
                )
                continue
            base = group[0]["offset"]
            end = group[-1]["offset"] + group[-1]["rows"]
            view = RowView(rows, range(base, end))
            timeline = compile_timeline(pipe, view, group, fragment)
            counts = write_timeline(pipe, cfg, out, timeline, group, index)
            index = counts["next_shard"]
            annotation = {
                worlds[s["piece"]["relpath"]]
                for s in group
                if s["piece"]["relpath"] in worlds
            }
            if len(annotation) > 1:
                raise ValueError("Continuous fragment crosses annotation sessions")
            stream = {
                "id": timeline["id"],
                "parent_fragment": fragment["fragment_id"],
                "world_group": timeline["world_group"],
                "annotation_world": next(
                    iter(annotation), identity(["orphan", timeline["id"]])
                ),
                "source_pieces": [s["piece"]["piece_id"] for s in group],
                "origin": timeline["origin"],
                "raw_rows": len(view),
                "kept_rows": len(timeline["kappa"]),
                "gui_cut_rows": sorted(timeline["gui"].cut),
                "excluded_grid": timeline["excluded"],
                **counts,
            }
            streams.append(stream)
            for source in group:
                for clip in by_source.get(source["piece"]["relpath"], []):
                    result = mapping.map_clip(
                        clip,
                        timeline,
                        source["offset"] - base,
                        pipe.clips.delta,
                        cfg.max_shift,
                    )
                    if result["status"] == "mapped":
                        info = clip_info(
                            result, timeline, pipe.clips.delta, cfg.out_dir
                        )
                        name = result["canonical_history_id"]
                        clips[name] = info
                        result.update(
                            {
                                key: info[key]
                                for key in ("classical_valid", "classical_reasons")
                            }
                        )
                        result["clip_ref"] = str(
                            shard_path(Path(cfg.out_dir) / "fragments", k)
                            / "clips"
                            / name
                        )
                    mapped.append(result)
        audit = [
            {k: v for k, v in s.items() if k not in ("validation", "target")}
            for s in sources
        ]
        summary = {
            "manifest_index": k,
            "parent_fragment": fragment["fragment_id"],
            "first_shard": first_shard,
            "next_shard": index,
            "sources": audit,
            "streams": streams,
        }
        with write_shard(out / "fragments", k) as staged:
            (staged / "summary.json").write_bytes(json_bytes(summary))
            (staged / "mappings.jsonl").write_bytes(jsonl(mapped))
            for name, info in clips.items():
                directory = staged / "clips" / name
                directory.mkdir(parents=True)
                (directory / "clip.json").write_bytes(json_bytes(info))
            seal(staged)
    finally:
        rows.close()
    temporary.unlink(missing_ok=True)
    cleanup(cfg, summary)
    return summary


def collection(cfg):
    source_cfg = load_source_config(REPO / cfg.perception_config)
    classifier, classifier_id = gamma_of(source_cfg)
    pipe = pipeline(source_cfg, classifier)
    out, root = REPO / cfg.out_dir, REPO / cfg.acquisition_plan
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        legacy, by_source = inputs.legacy_inputs(cfg, manifest)
        labels = inputs.label_inputs(cfg)
        worlds, sessions = inputs.annotation_worlds(cfg)
        head = inputs.header(cfg, pipe, classifier_id, labels, sessions)
        next_fragment = complete_shards(out / "fragments")
        first_shard, counts = 0, Counter()
        with tqdm(
            total=manifest["totals"]["fragments"],
            desc="Continuous fragments",
            unit="fragment",
            mininterval=10,
        ) as bar:
            for k, fragment, pieces in plans(root):
                if k < next_fragment:
                    summary = recover(cfg, out, k)
                else:
                    summary = process_fragment(
                        pipe,
                        cfg,
                        out,
                        k,
                        fragment,
                        pieces,
                        by_source,
                        worlds,
                        first_shard,
                        {
                            "classifier": classifier_id,
                            "source_config": head["fingerprint"]["source_config"],
                        },
                    )
                first_shard = summary["next_shard"]
                counts["sources"] += len(summary["sources"])
                counts["streams"] += len(summary["streams"])
                counts["rejected_sources"] += sum(
                    s["reason"] is not None for s in summary["sources"]
                )
                for stream in summary["streams"]:
                    counts.update(
                        {
                            key: stream[key]
                            for key in (
                                "observations",
                                "intervals",
                                "valid_intervals",
                                "frame_payload_bytes",
                            )
                        }
                    )
                write_json(
                    out / "run.json",
                    head | {"counts": dict(counts), "completed_fragments": k + 1},
                )
                bar.update()
        if (
            counts["sources"] != manifest["totals"]["pieces"]
            or complete_shards(out / "shards") != first_shard
        ):
            raise ValueError("Published source/shard counts do not reconcile")
        counts.update(
            mapping.finish(
                cfg, out, legacy, labels, load_split_config(REPO / cfg.split_config)
            )
        )
        write_json(
            out / "run.json",
            head
            | {
                "complete": True,
                "counts": dict(counts),
                "completed_fragments": manifest["totals"]["fragments"],
            },
        )
    finally:
        pipe.close()


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m mcpref.datasets.continuous.run")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        collection(load_config(args.config))
    except FAILURES as error:
        return failed(parser.prog, error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
