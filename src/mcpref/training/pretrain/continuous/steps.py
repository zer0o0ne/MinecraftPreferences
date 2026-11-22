"""Bounded K-observation unroll and its globally normalized losses."""

from mcpref.models.perception.continuous import observation_step
from mcpref.training.optim import update
from mcpref.training.pretrain.continuous.events import WindowResult, backward, notify
from mcpref.training.pretrain.continuous.losses.own import (
    action_input,
    eligible,
    phi_interval,
    policy_ce,
    policy_shift,
)
from mcpref.training.pretrain.continuous.losses.preference import endpoints
from mcpref.training.pretrain.continuous.state import detached


def factors(cfg, population: dict) -> dict:
    return {
        key: population["M"] * weight / population[name] if population[name] else 0.0
        for key, name, weight in (
            ("phi", "N_phi", cfg.w_phi),
            ("pi", "N_pi", cfg.w_pi),
            ("P", "N_P", cfg.w_P),
        )
    }


def own_losses(context, stream, observation, z, result):
    modules, own = context.modules, {}
    previous = stream.previous
    incoming = previous["interval"] if previous else None
    adjacent = bool(incoming and incoming["b"] == observation.id)
    if adjacent and incoming["valid"]:
        terms, diagnostics = phi_interval(modules["phi_loss"], previous, z, observation)
        own["phi"] = terms["total"]
        result.add("phi", terms)
        result.phi_rows.append(
            {key: value.detach() for key, value in diagnostics.items()}
        )
        result.counts["phi"] += 1
    if stream.pending is not None:
        pending = stream.pending
        if pending["target"] != observation.id:
            raise RuntimeError(
                "pending policy target does not match the next observation"
            )
        term, zero = policy_shift(modules["pi"], modules["pi_loss"], pending, z)
        own["pi"] = term
        result.add("pi", {"shift": term})
        result.shift_rows.append(
            {
                "id": pending["id"],
                "total": term.detach(),
                "zero_shift": zero.detach(),
                "valid": pending["valid"],
            }
        )
        result.counts["shift"] += 1
        stream.pending = None
    if not adjacent or not incoming["known"]:
        stream.policy = []
    pi = modules["pi"]
    stream.policy.append(
        {
            "id": observation.id,
            "z": z,
            "c": action_input(pi, incoming if adjacent else None, z),
        }
    )
    stream.policy = stream.policy[-pi.history :]
    if eligible(observation):
        ce, stream.pending, row = policy_ce(pi, stream, observation, z)
        own["pi"] = own.get("pi", 0) + ce
        result.add("pi", {"action": ce})
        result.ce_rows.append(row)
        result.counts["ce"] += 1
    return own


def window(context, stream, archive, observations, population, hooks=None):
    result, live = WindowResult(), {}
    weights = factors(context.cfg, population)
    preference_history = context.modules["P"].history if "P" in context.modules else 1
    for observation in observations:
        z, stream.recurrent = observation_step(
            context.modules["phi"],
            observation.frame,
            observation.line,
            stream.recurrent,
        )
        own = own_losses(context, stream, observation, z, result)
        for key, value in own.items():
            live[key] = live.get(key, 0) + weights[key] * value
        stream.remember(observation.id, z, preference_history)
        if observation.endpoints:
            endpoints(
                context, hooks, archive, stream, observation, result, weights["P"]
            )
        stream.previous = {
            "id": observation.id,
            "z": z,
            "frame": observation.frame,
            "meta": observation.meta,
            "interval": observation.interval,
        }
    backward(context, hooks, live, {"own": result}, retain=False)
    result.weighted.update({key: value.detach() for key, value in live.items()})
    notify(hooks, "before_update", context, result)
    result.lr = update(
        context.cfg, context.optimizer, context.groups, context.step, context.total
    )
    notify(hooks, "after_update", context, result)
    context.optimizer.zero_grad(set_to_none=True)
    return stream.detach(), detached(result, "cpu")
