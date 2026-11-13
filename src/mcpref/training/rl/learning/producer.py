"""Checkpoint-backed RL diagnostic observations and outbox handoff."""

import math

import torch

from mcpref.diagnostics.core.observations import UpdateObservations
from mcpref.diagnostics.core.runtime import DiagnosticRun
from mcpref.diagnostics.core.stats import distribution
from mcpref.diagnostics.core.types import ProducerContext, metric_record
from mcpref.diagnostics.producers.policy import on_ppo_observation, on_shift_observation
from mcpref.diagnostics.producers.update import on_update_observation
from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.shards.store import name
from mcpref.training.optim import iter_update_terms


def empty_observations() -> dict:
    return {
        "ppo": [],
        "shift": [],
        "logits": [],
        "ppo_clusters": [],
        "shift_clusters": [],
    }


def _cat(items, field):
    return torch.cat([getattr(item, field) for item in items])


def _context(cfg, cycle, t, producer, source, clusters):
    return ProducerContext(
        f"{producer}:{cycle:06d}",
        "train",
        "train",
        f"{cfg.out_dir}/{cycle:06d}",
        f"{cfg.out_dir}/{cycle:06d}",
        {},
        (t,),
        tuple(clusters),
        (source,),
    )


def _stage_ppo(items, cfg, diagnostics, context):
    if not items:
        return None
    first = items[0]
    observation = type(first)(
        _cat(items, "ratios"),
        _cat(items, "advantages"),
        _cat(items, "behavior"),
        _cat(items, "policy"),
        _cat(items, "actions"),
        first.clip,
        tuple(kind for item in items for kind in item.kinds),
        _cat(items, "eligible"),
    )
    return on_ppo_observation(observation, cfg=diagnostics.cfg, context=context)


def _stage_shift(items, cfg, diagnostics, context):
    if not items:
        return None
    first = items[0]
    observation = type(first)(
        _cat(items, "predicted"),
        _cat(items, "actual"),
        first.alpha,
        _cat(items, "distance"),
        _cat(items, "constraint"),
        _cat(items, "dual"),
        first.slack,
        _cat(items, "zero_skill"),
    )
    return on_shift_observation(observation, cfg=diagnostics.cfg, context=context)


@measured("diagnostics.gradients")
def capture_gradients(groups) -> dict:
    return {
        groups.parameter_names[id(parameter)]: parameter.grad.detach().clone()
        for parameters in groups.values()
        for parameter in parameters
        if parameter.grad is not None
    }


def _update_observation(cfg, observations, gradients, groups, optimizer, lr, t):
    names = {
        group: tuple(groups.parameter_names[id(p)] for p in parameters)
        for group, parameters in groups.items()
    }
    gradients = {
        name: value
        for group in names.values()
        for name, value in gradients.items()
        if name in group
    }
    norm = math.sqrt(
        sum(
            float(
                torch.linalg.vector_norm(
                    value, ord=2, dtype=torch.promote_types(value.dtype, torch.float32)
                )
            )
            ** 2
            for value in gradients.values()
        )
    )
    updates = {
        term.parameter: None if term.delta is None else term.delta.detach().clone()
        for term in iter_update_terms(cfg, optimizer, groups, lr)
    }
    action_count = observations["action_count"]
    behavior = _cat(observations["ppo"], "behavior") if observations["ppo"] else None
    supported = (
        (behavior > 0).any(dim=0)
        if behavior is not None
        else torch.zeros(action_count, dtype=torch.bool)
    )
    logits = (
        torch.cat(observations["logits"])
        if observations["logits"]
        else torch.empty(0, action_count)
    )
    return UpdateObservations(
        gradients,
        updates,
        names,
        min(1, cfg.grad_clip / (norm + 1e-6)),
        next(iter(lr.values())),
        t,
        logits,
        supported,
    )


def register(
    diagnostics: DiagnosticRun, cfg, cycle: int, start: int, restored: int
) -> None:
    event = start + restored
    watermark = diagnostics.state["channels"].get("rl", {}).get("watermark", event)
    diagnostics.register("rl", watermark=watermark, next=event)
    diagnostics.recover("rl")


@measured("diagnostics.record")
def stage(
    diagnostics: DiagnosticRun,
    cfg,
    cycle: int,
    start: int,
    t: int,
    j: int,
    observations: dict,
    gradients: dict | None,
    groups,
    optimizer,
    lr: dict,
    entry: dict,
) -> None:
    source = f"{cfg.out_dir}/{cycle:06d}/checkpoints/{name(j)}/losses.json"
    records = [
        _stage_ppo(
            observations["ppo"],
            cfg,
            diagnostics,
            _context(cfg, cycle, t, "ppo", source, observations["ppo_clusters"]),
        ),
        _stage_shift(
            observations["shift"],
            cfg,
            diagnostics,
            _context(cfg, cycle, t, "shift", source, observations["shift_clusters"]),
        ),
    ]
    actions = observations["action_count"]
    context = _context(
        cfg, cycle, t, "update", source, [f"action:{a}" for a in range(actions)]
    )
    if cfg.gradient_diagnostics:
        update = _update_observation(
            cfg, observations, gradients, groups, optimizer, lr, t
        )
        update_record = on_update_observation(
            update, cfg=diagnostics.cfg, context=context
        )
    else:
        logits = (
            torch.cat(observations["logits"]).to(torch.float64)
            if observations["logits"]
            else torch.empty(0, actions, dtype=torch.float64)
        )
        update_record = metric_record(
            "rl",
            f"{context.producer}:{t}",
            context,
            {
                "update.learning_rate": next(iter(lr.values())),
                "update.schedule_position": t,
                "update.logit_norm": distribution(
                    torch.linalg.vector_norm(logits, ord=2, dim=-1).cpu().numpy(),
                    diagnostics.cfg,
                ),
            },
            {"update.logit_norm": list(context.cluster_ids)},
        )
    update_record["joined"]["outputs"] = entry
    records.append(update_record)
    event = start + t
    for offset, record in enumerate(record for record in records if record is not None):
        diagnostics.stage(
            "rl", event * 4 + offset, record["id"], event + 1, [source], record
        )


@measured("diagnostics.publish")
def accept(diagnostics: DiagnosticRun, cfg, cycle: int, start: int, t: int) -> None:
    diagnostics.recover("rl")
    event = start + t
    diagnostics.finish(
        "rl",
        source_complete=True,
        done=diagnostics.state["channels"]["rl"]["next"] >= event + 1,
    )
