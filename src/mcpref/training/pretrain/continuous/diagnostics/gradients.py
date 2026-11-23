"""Gradients of the actual truncated graph, collected before its real backward."""

import shutil

from mcpref.infrastructure.runs.files import json_bytes, write_json
from mcpref.infrastructure.shards.store import name
from mcpref.training.pretrain.continuous.diagnostics.selection import gradient_steps
from mcpref.training.pretrain.gradients.collect import collect_micro
from mcpref.training.pretrain.gradients.work import reduce_work
from mcpref.training.pretrain.update import update_and_align


class Gradients:
    def __init__(self, context):
        self.root = context.root / "gradient-flow"
        self.selected = gradient_steps(context)
        self.current = None
        self.micro = 0
        self.pending = []
        shutil.rmtree(self.root / "work", ignore_errors=True)
        write_json(
            self.root / "selection.json",
            {
                "mode": "continuous",
                "steps": [
                    {"step": step, "roles": roles}
                    for step, roles in self.selected.items()
                ],
            },
        )

    def collect(self, context, components):
        if context.step not in self.selected:
            return
        if self.current != context.step:
            self.current, self.micro = context.step, 0
        work = self.root / "work" / name(context.step)
        collect_micro(
            dict.fromkeys(("phi", "pi", "P")) | components,
            context.modules,
            context.groups,
            work,
            self.micro,
        )
        self.micro += 1

    def finish_step(self, context, result):
        if context.step not in self.selected:
            return
        work = self.root / "work" / name(context.step)
        reduced = reduce_work(work, groups=context.groups)
        result.updates, alignment = update_and_align(
            context.cfg, context.optimizer, context.groups, result.lr, work
        )
        self.pending.append(
            {
                "version": 1,
                "mode": "continuous",
                "step": context.step,
                "roles": self.selected[context.step],
                "counts": dict(result.counts),
                "evidence": {
                    "kind": "real_pretrain_step",
                    "gradient_path": "sliding_replay_and_detached_earlier_endpoints",
                },
                **reduced,
                "update_alignment": alignment,
            }
        )
        shutil.rmtree(work)

    def checkpoint(self, staged):
        for record in self.pending:
            directory = staged / "gradient_flow" / name(record["step"])
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_bytes(json_bytes(record))
        self.pending = []
