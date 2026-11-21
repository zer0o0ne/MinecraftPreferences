"""Atomic continuous reference artifacts and producer-owned outboxes."""

import json
from dataclasses import asdict

import torch

from mcpref.config import REPO
from mcpref.diagnostics.core.artifacts import write_pairs
from mcpref.diagnostics.core.types import ProducerContext
from mcpref.diagnostics.preference.suites import GlobalFit, Stratum
from mcpref.diagnostics.producers.preference import join_reference, on_reference
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import (
    complete_shards,
    name,
    shard_path,
    write_shard,
)
from mcpref.training.pretrain.continuous.diagnostics.records import endpoint_source
from mcpref.training.pretrain.continuous.diagnostics.reference import (
    evaluate_initial,
    evaluate_reference,
    select_reference,
)


class References:
    def __init__(self, context, diagnostics, channel):
        self.context, self.diagnostics, self.channel = context, diagnostics, channel
        self.directory = context.root / "reference"
        self.fit = self.selection = self.initial = None

    def start(self):
        context = self.context
        if not context.index.run["populations"]["train"]["N_P"]:
            return
        count = complete_shards(self.directory)
        if not count:
            if context.step:
                raise RuntimeError(
                    "Initial continuous reference is missing after training started"
                )
            self.selection, self.fit = select_reference(context.cfg, context.index)
            initial, metrics, pairs, observations = evaluate_initial(
                context.cfg, context.modules, context.index, self.selection, self.fit
            )
            self.initial = initial
            self.publish(0, metrics, pairs, observations, initial)
        else:
            root = shard_path(self.directory, 0)
            self.selection = json.loads((root / "selection.json").read_bytes())
            raw = json.loads((root / "baselines.json").read_bytes())
            self.fit = GlobalFit(
                **raw | {"strata": tuple(Stratum(**s) for s in raw["strata"])}
            )
            self.initial = torch.load(
                root / "initial.pt", map_location="cpu", weights_only=False
            )
        self.due()

    def publish(self, ordinal, metrics, pairs, observations, initial=None):
        context = self.context
        root = self.directory / name(ordinal)
        sources = [
            str((root / file).relative_to(REPO))
            for file in ("metrics.json", "pairs.json")
        ]
        i = ordinal * 8
        for part in ("train", "validation"):
            chosen = [
                (j, edge)
                for j, edge in enumerate(self.selection["items"])
                if edge["part"] == part
            ]
            by_index = {pair.micro_batch: pair for pair in pairs}
            for condition in ("joint", "P_only", "phi_only", "initial"):
                producer = ProducerContext(
                    "pretrain-continuous-reference",
                    part,
                    condition,
                    "pretrain",
                    str(context.root.relative_to(REPO)),
                    {"phase": context.phase},
                    (metrics["step"],),
                    tuple(
                        endpoint_source(context.index, edge["a"]) for _, edge in chosen
                    ),
                    tuple(sources),
                )
                record = on_reference(
                    observations[condition][part],
                    pairs=tuple(by_index[j] for j, _ in chosen),
                    cfg=self.diagnostics.cfg,
                    context=producer,
                )
                record["id"] = f"continuous:{context.phase}:{record['id']}"
                self.diagnostics.stage(
                    self.channel,
                    i,
                    f"reference:{ordinal}",
                    ordinal + 1,
                    sources,
                    record,
                )
                i += 1
        with write_shard(self.directory, ordinal) as staged:
            (staged / "metrics.json").write_bytes(json_bytes(metrics))
            write_pairs(staged, pairs)
            if initial is not None:
                (staged / "selection.json").write_bytes(json_bytes(self.selection))
                (staged / "baselines.json").write_bytes(json_bytes(asdict(self.fit)))
                torch.save(initial, staged / "initial.pt")
        self.diagnostics.accept(
            self.channel, f"reference:{ordinal}", join=join_reference
        )

    def due(self):
        context = self.context
        if self.initial is None:
            return
        count = complete_shards(self.directory)
        expected = context.step // context.cfg.reference_steps + 1
        if count == expected:
            return
        if count != expected - 1 or context.step % context.cfg.reference_steps:
            raise RuntimeError("Continuous references do not match the restored step")
        metrics, pairs, observations = evaluate_reference(
            context.cfg,
            context.modules,
            context.index,
            self.selection,
            self.fit,
            self.initial,
            context.step,
        )
        self.publish(expected - 1, metrics, pairs, observations)
