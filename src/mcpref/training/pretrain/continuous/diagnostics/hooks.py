"""Ordinary and reference diagnostics committed with continuous updates."""

import sys
from dataclasses import replace

from mcpref.config import REPO
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.diagnostics.producers.preference import join_reference
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.training.pretrain.continuous.diagnostics.gradients import Gradients
from mcpref.training.pretrain.continuous.diagnostics.records import (
    decorate,
    join_step,
    stage_step,
    window_metrics,
)
from mcpref.training.pretrain.continuous.diagnostics.reference_store import References
from mcpref.training.pretrain.continuous.state import detached
from mcpref.training.pretrain.update import gradient_stats, update_stats


def report_start(context):
    cfg = context.cfg
    population = context.index.run["populations"]["train"]
    count = population["observations"]
    updates = (count + cfg.batch_size - 1) // cfg.batch_size
    print(
        f"Continuous {context.phase}: {count} training observations/epoch, "
        f"{population['fragments']} fragments, {updates} updates/epoch; "
        f"{context.step}/{context.total} total updates completed.",
        file=sys.stderr,
    )
    print(
        f"Continuous settings: B={cfg.batch_size}, K={cfg.tbptt_states}, "
        f"precision={cfg.precision}, activation_checkpointing={cfg.activation_checkpointing}; "
        f"checkpoint_steps={cfg.checkpoint_steps}, diagnostic_window={cfg.diagnostic_window}, "
        f"reference_steps={cfg.reference_steps}; "
        f"gradient_diagnostics={cfg.gradient_diagnostics}, "
        f"publish_diagnostics={cfg.publish_diagnostics}. "
        "Reference evaluation is independent of both diagnostic flags.",
        file=sys.stderr,
    )


class Diagnostics:
    def start(self, context):
        report_start(context)
        self.pending = []
        self.diagnostics = (
            DiagnosticRun.load(REPO / context.cfg.diagnostics)
            if context.cfg.publish_diagnostics
            else None
        )
        prefix = f"pretrain-continuous-{context.phase}"
        self.step_channel, self.reference_channel = (
            f"{prefix}-step",
            f"{prefix}-reference",
        )
        for channel, watermark, join in (
            (self.step_channel, 1, join_step),
            (self.reference_channel, 0, join_reference),
        ):
            if self.diagnostics is not None:
                self.diagnostics.register(channel, watermark=watermark, next=watermark)
                self.diagnostics.recover(channel, join=join)
        self.reference = References(context, self.diagnostics, self.reference_channel)
        self.reference.start()
        self.gradients = (
            Gradients(context) if context.cfg.gradient_diagnostics else None
        )

    def before_backward(self, context, components, events):
        if self.gradients is not None:
            self.gradients.collect(context, components)

    def before_update(self, context, result):
        if self.gradients is not None:
            result.gradients = gradient_stats(
                context.groups,
                context.cfg.grad_clip,
                preference=context.modules.get("P"),
            )

    def after_update(self, context, result):
        if self.gradients is not None:
            result.updates = update_stats(
                context.cfg, context.optimizer, context.groups, result.lr
            )
            self.gradients.finish_step(context, result)
        result = detached(result, "cpu")
        record, rows = decorate(context, result, self.reference.fit)
        self.pending.append((context.step, result, record, rows))

    def checkpoint(self, context, staged, results):
        if len(results) != len(self.pending):
            raise RuntimeError("Continuous diagnostics lost a training update")
        windows, buffer = [], []
        for step, result, _, _ in self.pending:
            buffer.append((step, result))
            if step % context.cfg.diagnostic_window == 0 or step == context.total:
                windows.append(window_metrics(context, buffer, self.reference.fit))
                buffer = []
        if buffer:
            raise RuntimeError("Continuous checkpoint splits a diagnostic window")
        (staged / "losses.json").write_bytes(
            json_bytes([item[2] for item in self.pending])
        )
        (staged / "diagnostics.json").write_bytes(json_bytes(windows))
        final = staged.with_name(staged.name.removesuffix(".part"))
        for step, result, _, rows in (
            self.pending if self.diagnostics is not None else ()
        ):
            sources = [str((final / "losses.json").relative_to(REPO))]
            if step % context.cfg.diagnostic_window == 0 or step == context.total:
                sources.append(str((final / "diagnostics.json").relative_to(REPO)))
            stage_step(
                self.diagnostics,
                replace(context, step=step),
                self.reference.fit,
                result,
                rows,
                sources,
                self.step_channel,
            )
        if self.gradients is not None:
            self.gradients.checkpoint(staged)
        self.pending = []

    def published(self, context, ordinal):
        if self.diagnostics is not None:
            self.diagnostics.recover(self.step_channel, join=join_step)
        self.reference.due()

    def finish(self, context):
        if self.diagnostics is None:
            return
        for channel in (self.step_channel, self.reference_channel):
            self.diagnostics.finish(channel, source_complete=True, done=True)
