"""Raw-pair counterfactual evaluation of specs/pretrain-diagnostics.md (27)-(28)."""

import torch

from mcpref.diagnostics.preference.metrics import quantile
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.data.items import clip_reader
from mcpref.training.pretrain.references.eval import evaluation_state


def labelled_items(batch: list[dict], micro_batch: int) -> list[tuple[int, int, dict]]:
    result = []
    for start in range(0, len(batch), micro_batch):
        pair = 0
        for item in batch[start : start + micro_batch]:
            if "labels" not in item:
                continue
            result.append((start // micro_batch, pair, item))
            pair += 1
    return result


def preference_total(
    modules: Modules, Z: torch.Tensor, labels: dict, preference=None
) -> float:
    network = modules["P"] if preference is None else preference
    out = network(Z[:1], Z[1:])
    return modules["P_loss"].terms(out, [labels])["total"][0].item()


def baseline_pairs(
    cfg: PretrainConfig,
    modules: Modules,
    items: list[dict],
    run: dict,
    progress=None,
) -> list[float]:
    """(27): pre-update eval-mode preference loss for each raw labelled pair."""
    result = []
    read = clip_reader(cfg, run)
    with evaluation_state(cfg, modules):
        for item in items:
            Z = modules["phi"]([read(item["a"]), read(item["b"])])
            result.append(preference_total(modules, Z, item["labels"]))
            if progress is not None:
                progress.update()
                progress.refresh()
    return result


def post_step_pairs(
    cfg: PretrainConfig,
    modules: Modules,
    frozen_P: torch.nn.Module,
    items: list[dict],
    run: dict,
    progress=None,
) -> tuple[list[float], list[float]]:
    """(27): one updated phi call and frozen/current preference calls per pair."""
    only_phi, joint = [], []
    read = clip_reader(cfg, run)
    frozen_P.eval()
    with evaluation_state(cfg, modules):
        for item in items:
            Z = modules["phi"]([read(item["a"]), read(item["b"])])
            only_phi.append(preference_total(modules, Z, item["labels"], frozen_P))
            joint.append(preference_total(modules, Z, item["labels"]))
            if progress is not None:
                progress.update()
                progress.refresh()
    return only_phi, joint


def pair_distribution(values: list[float], quantiles: tuple[float, ...]) -> dict:
    count = len(values)
    return {
        "count": count,
        "mean": sum(values) / count if count else None,
        "median": quantile(values, 0.5),
        "quantiles": {str(q): quantile(values, q) for q in quantiles},
        "harmful": sum(value > 0 for value in values) / count if count else None,
        "helpful": sum(value < 0 for value in values) / count if count else None,
        "zero": sum(value == 0 for value in values) / count if count else None,
    }


def counterfactual_metrics(pairs: list[dict], quantiles: tuple[float, ...]) -> dict:
    return {
        key: pair_distribution([pair[key] for pair in pairs], quantiles)
        for key in ("delta_only_phi", "delta_joint")
    }


def pair_records(
    labelled: list[tuple[int, int, dict]],
    baseline: list[float],
    only_phi: list[float],
    joint: list[float],
) -> list[dict]:
    return [
        {
            "micro_batch": micro,
            "pair_in_micro": pair,
            "a": item["a"],
            "b": item["b"],
            "labels": item["labels"],
            "baseline": before,
            "only_phi_post": after_phi,
            "joint_post": after_joint,
            "delta_only_phi": after_phi - before,
            "delta_joint": after_joint - before,
        }
        for (micro, pair, item), before, after_phi, after_joint in zip(
            labelled, baseline, only_phi, joint
        )
    ]
