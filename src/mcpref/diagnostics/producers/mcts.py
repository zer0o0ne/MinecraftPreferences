"""Completed StepRecord policy agreement, exploration and realized outcomes."""

import math

import numpy as np
import torch

from mcpref.diagnostics.core.config import DiagnosticsConfig
from mcpref.diagnostics.core.stats import distribution, rate
from mcpref.diagnostics.core.types import ProducerContext, metric_record
from mcpref.runtime.mcts.tree import quantize
from mcpref.runtime.sampling.shards import StepRecord


def _rank(values: np.ndarray, action: int) -> int:
    return 1 + int(np.sum(values > values[action]))


def _kl(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.any(right[left > 0] <= 0):
        return None
    selected = left > 0
    return float(
        np.sum(left[selected] * (np.log(left[selected]) - np.log(right[selected])))
    )


def _js(left: np.ndarray, right: np.ndarray) -> float:
    middle = (left + right) / 2
    first, second = _kl(left, middle), _kl(right, middle)
    return (first + second) / 2


def _binary(value: bool, context: ProducerContext, cfg: DiagnosticsConfig):
    return rate([float(value)], list(context.cluster_ids), cfg)


def on_step_record(
    record: StepRecord, *, cfg: DiagnosticsConfig, context: ProducerContext
):
    """(11), (13)-(14), (16): no simulations or fresh model evaluation."""
    root = record.tree["root"]
    children = root["children"]
    visits = record.probabilities.detach().cpu().numpy().astype(np.float64)
    try:
        policy = record.policy_probabilities.detach().cpu().numpy().astype(np.float64)
    except RuntimeError as error:
        if "unavailable" not in str(error):
            raise
        return metric_record(
            "mcts", f"step:{record.index}", context, {"mcts.unavailable": 1}, {}
        ) | {"joined": {"reason": "root_policy_unavailable"}}
    if len(visits) != len(policy) or not children:
        raise ValueError(
            "a StepRecord needs aligned full probabilities and root children"
        )
    actions = np.asarray([child["action"] for child in children], dtype=int)
    if (
        record.action < 0
        or record.action >= len(visits)
        or np.any((actions < 0) | (actions >= len(visits)))
    ):
        return metric_record(
            "mcts", f"step:{record.index}", context, {"mcts.unavailable": 1}, {}
        ) | {"joined": {"reason": "inconsistent_root_action_indices"}}
    priors = np.asarray([child["prior"] for child in children], dtype=np.float64)
    order = sorted(range(len(children)), key=lambda i: (-priors[i], int(actions[i])))
    prior_full = np.zeros_like(visits)
    prior_full[actions] = priors
    normalized_prior = prior_full / prior_full.sum()
    alpha = torch.tensor(context.filters["alpha"], dtype=record.delta.dtype)
    epsilon = context.filters["epsilon"]
    z_real = root["z"] + record.delta
    matches = [
        torch.equal(
            quantize(child["x"] - z_real, alpha, epsilon), torch.zeros_like(child["x"])
        )
        for child in children
    ]
    chosen_hit = torch.equal(
        quantize(record.predicted - record.delta, alpha, epsilon),
        torch.zeros_like(record.delta),
    )
    width = context.filters["width"]
    top = order[:width]
    matched_mass = float(
        sum(visits[actions[i]] for i, hit in enumerate(matches) if hit)
    )
    kl = _kl(normalized_prior[actions], visits[actions])
    metrics = {
        "mcts.top1_prior_agreement": _binary(
            int(prior_full.argmax()) == int(visits.argmax()), context, cfg
        ),
        "mcts.top1_policy_agreement": _binary(
            int(policy.argmax()) == int(visits.argmax()), context, cfg
        ),
        "mcts.rank_prior": distribution([_rank(prior_full, record.action)], cfg),
        "mcts.rank_visits": distribution([_rank(visits, record.action)], cfg),
        "mcts.policy_support_mass": distribution([policy[actions].sum()], cfg),
        "mcts.kl_prior_visits": distribution(
            [] if kl is None else [kl],
            cfg,
            missing=int(kl is None),
            reason="zero_visit_support" if kl is None else None,
        ),
        "mcts.js_prior_visits": distribution(
            [_js(normalized_prior[actions], visits[actions])], cfg
        ),
        "mcts.visit_entropy": distribution(
            [-sum(value * math.log(value) for value in visits if value > 0)], cfg
        ),
        "mcts.top_width_prior_mass": distribution([priors[top].sum()], cfg),
        "mcts.chosen_hit": _binary(chosen_hit, context, cfg),
        "mcts.top1_hit": _binary(matches[order[0]], context, cfg),
        "mcts.topk_hit": _binary(any(matches[i] for i in top), context, cfg),
        "mcts.support_hit": _binary(any(matches), context, cfg),
        "mcts.matched_mass": distribution([matched_mass], cfg),
        "mcts.matched_hit": _binary(matched_mass > 0, context, cfg),
        "mcts.d_alpha_error": distribution(
            [
                float(
                    torch.linalg.vector_norm(
                        alpha * (record.predicted - record.delta), ord=2
                    )
                )
            ],
            cfg,
        ),
        "mcts.zero_shift_error": distribution(
            [float(torch.linalg.vector_norm(alpha * record.delta, ord=2))], cfg
        ),
    }
    metrics["mcts.kl_prior_visits_undefined"] = int(kl is None)
    mask = torch.as_tensor(record.before.mask).numpy().astype(bool)
    allowed_probability = float(policy[mask].sum())
    normalized = (
        policy[mask] / allowed_probability if allowed_probability > 0 else np.array([])
    )
    metrics |= {
        "mask.count": distribution([int(mask.sum())], cfg),
        "mask.empty": _binary(not mask.any(), context, cfg),
        "mask.singleton": _binary(mask.sum() == 1, context, cfg),
        "mask.probability": distribution([1 - allowed_probability], cfg),
        "mask.allowed_entropy": distribution(
            []
            if allowed_probability == 0
            else [-sum(value * math.log(value) for value in normalized if value > 0)],
            cfg,
            missing=int(allowed_probability == 0),
            reason="zero_allowed_probability" if allowed_probability == 0 else None,
        ),
        "mask.executed_violation": _binary(not mask[record.action], context, cfg),
    }
    populations = {name: list(context.cluster_ids) for name in metrics}
    joined = {"matched_reliability": [{"mass": matched_mass, "hit": any(matches)}]}
    return metric_record(
        "mcts", f"step:{record.index}", context, metrics, populations
    ) | {"joined": joined}
