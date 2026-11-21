"""Shared reference reporting over genuine continuous prefixes, never isolated clips."""

import copy

from mcpref.diagnostics.core.types import ReferencePair
from mcpref.diagnostics.preference.suites import pool_preferences
from mcpref.training.checkpoints import cpu_states
from mcpref.training.pretrain.continuous.diagnostics.reference_replay import (
    evaluation_state,
    replay,
)
from mcpref.training.pretrain.continuous.diagnostics.reference_selection import (
    select_reference,
)
from mcpref.training.pretrain.references.metrics import (
    preference_data,
    preference_part,
)

__all__ = ["evaluate_initial", "evaluate_reference", "select_reference"]
CONDITIONS = ("joint", "P_only", "phi_only")
PARTS = ("train", "validation")


def _score(cfg, modules, selection, histories, initial):
    parts = {
        condition: {part: {"terms": [], "rows": []} for part in PARTS}
        for condition in CONDITIONS
    }
    frozen = None
    if initial is not None:
        frozen = copy.deepcopy(modules["P"])
        frozen.load_state_dict(initial["P0"], strict=True)
        frozen.eval()
    pairs = []
    for i, edge in enumerate(selection["items"]):
        A, B = (histories[edge[side]].to(cfg.device)[None] for side in ("a", "b"))
        joint = preference_data(modules, modules["P"](A, B), edge["labels"])
        scores = dict.fromkeys(CONDITIONS, joint)
        if initial is not None:
            old_a, old_b = (
                initial["Z0"][edge[side]].to(cfg.device)[None] for side in ("a", "b")
            )
            scores["P_only"] = preference_data(
                modules, modules["P"](old_a, old_b), edge["labels"]
            )
            scores["phi_only"] = preference_data(modules, frozen(A, B), edge["labels"])
        for condition, (terms, rows) in scores.items():
            parts[condition][edge["part"]]["terms"].append(terms)
            parts[condition][edge["part"]]["rows"].append(rows)
        baseline = (
            joint[0]["total"] if initial is None else initial["pairs"][i].baseline
        )
        phi = scores["phi_only"][0]["total"]
        pairs.append(
            ReferencePair(
                i,
                0,
                edge.get("original_a", edge["a"]),
                edge.get("original_b", edge["b"]),
                edge["labels"],
                baseline,
                phi,
                joint[0]["total"],
                phi - baseline,
                joint[0]["total"] - baseline,
            )
        )
    return parts, pairs


def _evaluate(cfg, modules, index, selection, fit, initial, step):
    with evaluation_state(cfg, modules):
        histories, own = replay(cfg, modules, index, selection)
        scores, pairs = _score(cfg, modules, selection, histories, initial)
        conditions = {
            condition: {
                part: preference_part(cfg, fit, scores[condition][part])
                for part in PARTS
            }
            for condition in CONDITIONS
        }
        conditions["initial"] = copy.deepcopy(
            conditions["phi_only"]
            if initial is None
            else initial["metrics"]["conditions"]["initial"]
        )
        for part in PARTS:
            joint = conditions["joint"][part]
            joint["losses"].update(
                {
                    key: value
                    for key, value in own[part]["losses"].items()
                    if not key.startswith("P")
                }
            )
            joint.update(
                {key: value for key, value in own[part].items() if key != "losses"}
            )
            joint["counts"]["P"] = len(scores["joint"][part]["terms"])
        observations = {
            condition: {
                part: pool_preferences(scores[condition][part]["rows"])
                for part in PARTS
            }
            for condition in CONDITIONS
        }
        observations["initial"] = (
            copy.deepcopy(observations["joint"])
            if initial is None
            else dict.fromkeys(PARTS)
        )
        metrics = {
            "version": 1,
            "mode": "continuous",
            "step": step,
            "conditions": conditions,
        }
        saved = (
            None
            if initial is not None
            else {
                "Z0": histories,
                "P0": cpu_states({"P": modules["P"]})["P"],
                "metrics": metrics,
                "pairs": pairs,
            }
        )
    return saved, metrics, pairs, observations


def evaluate_initial(cfg, modules, index, selection, fit):
    return _evaluate(cfg, modules, index, selection, fit, None, 0)


def evaluate_reference(cfg, modules, index, selection, fit, initial, step):
    _, metrics, pairs, observations = _evaluate(
        cfg, modules, index, selection, fit, initial, step
    )
    return metrics, pairs, observations
