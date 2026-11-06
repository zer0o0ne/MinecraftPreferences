"""Original pretrain losses, ordinary observations and selected component work."""

from collections.abc import Callable
from pathlib import Path

import torch

from mcpref.pretrain.checks import Groups
from mcpref.pretrain.clip import Clip
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.gradient_work import collect_micro
from mcpref.pretrain.micro_loss import MicroResult, Pairs, micro_loss
from mcpref.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
    preference_observation,
    step_observation,
)


def clips_and_pairs(items: list[dict]) -> tuple[list[str], Pairs]:
    """Notation K(B), R(B): the clips' paths in item order (a, then b), the pairs."""
    paths, pairs = [], []
    for item in items:
        if "clip" in item:
            paths.append(item["clip"])
        else:
            pairs.append((len(paths), len(paths) + 1, item["labels"]))
            paths += [item["a"], item["b"]]
    return paths, pairs


def add_components(totals: dict, result: MicroResult) -> None:
    for name in totals:
        if result.components[name] is not None:
            totals[name] = {
                key: totals[name][key] + value
                for key, value in result.components[name].items()
            }


def component_means(totals: dict, clips: int, pairs: int) -> dict:
    return {
        "phi": {key: value / clips for key, value in totals["phi"].items()},
        "pi": {key: value / clips for key, value in totals["pi"].items()},
        "P": (
            {key: value / pairs for key, value in totals["P"].items()}
            if pairs
            else None
        ),
    }


def step_counts(
    items: list[dict], clips: int, pairs: int, sum_p: float, observations
) -> dict:
    comparable = (
        int((observations.preference.inc == 0).sum().item())
        if observations.preference is not None
        else 0
    )
    return {
        "items": len(items),
        "clips": clips,
        "pairs": pairs,
        "comparable": comparable,
        "incomparable": pairs - comparable,
        "sum_p": sum_p if comparable else None,
        "action_targets": observations.policy.action_valid,
        "shift_targets": observations.policy.shift_valid,
        "phi_changed": observations.phi.changed,
        "phi_state_pairs": observations.phi.state_pairs,
    }


def gradient_steps(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_pi: float,
    w_P: float,
    micro_batch: int,
    cfg: PretrainConfig,
    selected: tuple[Groups, Path, object] | None,
) -> tuple[dict, list[dict]]:
    factors = (m / k_tot, w_pi, m * w_P / r_tot if r_tot else 0.0)
    totals, n_k, n_r = [0.0] * 4, 0, 0
    component_totals = {
        "phi": dict.fromkeys(("push", "constraint", "dual", "norm"), 0.0),
        "pi": dict.fromkeys(("action", "shift"), 0.0),
        "P": dict.fromkeys(("inc", "succ"), 0.0),
    }
    micro, sum_p, work_records = [], 0.0, []
    for j in range(0, len(items), micro_batch):
        paths, pairs = clips_and_pairs(items[j : j + micro_batch])
        loss, result = micro_loss(modules, [read(p) for p in paths], pairs, factors)
        if selected is None:
            loss.backward()
        else:
            groups, directory, progress = selected
            loss.backward(retain_graph=True)
            work_records.append(
                collect_micro(
                    result.weighted, modules, groups, directory, j // micro_batch
                )
            )
            if progress is not None:
                progress.update()
                progress.refresh()
        policy, state_rows = policy_observation(
            result.policy, cfg.calibration_bins, cfg.action_copy_eps
        )
        micro.append(
            MicroObservations(
                preference_observation(result.preference),
                policy,
                phi_observation(result.phi),
                state_rows,
            )
        )
        totals = [a + b for a, b in zip(totals, [*result.sums, loss.detach().item()])]
        add_components(component_totals, result)
        sum_p += sum(pair[2]["p"] for pair in pairs if pair[2]["inc"] == 0)
        n_k, n_r = n_k + len(paths), n_r + len(pairs)
        del loss, result
    phi, pi, P, loss = totals
    observations = step_observation(micro, n_k)
    logged = {
        "phi": phi / n_k,
        "pi": pi / n_k,
        "P": P / n_r if n_r else None,
        "loss": loss,
        "counts": step_counts(items, n_k, n_r, sum_p, observations),
        "components": component_means(component_totals, n_k, n_r),
        "observations": observations,
        "dual_lambda": modules["phi_loss"].log_lambda.exp().detach().item(),
    }
    return logged, work_records


def step_gradients(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_pi: float,
    w_P: float,
    micro_batch: int,
    *,
    cfg: PretrainConfig,
) -> dict:
    """Original ordered backwards, each followed by bounded CPU observation transfer."""
    return gradient_steps(
        items, modules, read, m, k_tot, r_tot, w_pi, w_P, micro_batch, cfg, None
    )[0]


def selected_step_gradients(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_pi: float,
    w_P: float,
    micro_batch: int,
    *,
    cfg: PretrainConfig,
    groups: Groups,
    directory: Path,
    progress=None,
) -> tuple[dict, list[dict]]:
    """Selected original backwards with retained graphs and atomic component work."""
    return gradient_steps(
        items,
        modules,
        read,
        m,
        k_tot,
        r_tot,
        w_pi,
        w_P,
        micro_batch,
        cfg,
        (groups, directory, progress),
    )
