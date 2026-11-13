"""Original pretrain losses, ordinary observations and selected component work."""

from collections.abc import Callable
from contextlib import closing, nullcontext
from pathlib import Path

import torch

from mcpref.datasets.clips.io import Clip, prefetch_clips
from mcpref.diagnostics.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
    preference_observation,
    raw_policy_observation,
    step_observation,
)
from mcpref.infrastructure.profiling.scopes import phase
from mcpref.training.groups import Groups
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.gradients.work import collect_micro
from mcpref.training.pretrain.steps.micro_loss import MicroResult, Pairs, micro_loss

COMPONENT_KEYS = {
    "phi": ("push", "constraint", "dual", "norm"),
    "pi": ("action", "shift"),
    "P": ("inc", "succ"),
}


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
        int((observations.preference.inc == 0).sum().tolist())
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


def logging_row(result: MicroResult, loss: torch.Tensor) -> list[torch.Tensor]:
    zero = loss.detach().new_zeros(())
    preference = result.components["P"]
    return [
        *result.sums,
        loss.detach(),
        *(result.components["phi"][key] for key in COMPONENT_KEYS["phi"]),
        *(result.components["pi"][key] for key in COMPONENT_KEYS["pi"]),
        *(
            preference[key] if preference is not None else zero
            for key in COMPONENT_KEYS["P"]
        ),
    ]


def finish_scalars(
    rows: list[list[torch.Tensor]], p_present: list[bool], dual: torch.Tensor
) -> tuple[list[float], dict, float]:
    values = torch.stack([torch.stack([*row, dual]) for row in rows]).cpu().tolist()
    totals = [0.0] * 4
    components = {
        name: dict.fromkeys(keys, 0.0) for name, keys in COMPONENT_KEYS.items()
    }
    offsets = {"phi": 4, "pi": 8, "P": 10}
    for j, row in enumerate(values):
        totals = [a + b for a, b in zip(totals, row[:4])]
        for name, keys in COMPONENT_KEYS.items():
            if name == "P" and not p_present[j]:
                continue
            offset = offsets[name]
            components[name] = {
                key: components[name][key] + row[offset + i]
                for i, key in enumerate(keys)
            }
    return totals, components, values[0][-1]


def gradient_steps(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_phi: float,
    w_pi: float,
    w_P: float,
    micro_batch: int,
    cfg: PretrainConfig,
    selected: tuple[Groups, Path, object] | None,
) -> tuple[dict, list[dict]]:
    factors = (m / k_tot, w_phi, w_pi, m * w_P / r_tot if r_tot else 0.0)
    n_k, n_r = 0, 0
    micro, sum_p, work_records = [], 0.0, []
    scalar_rows, p_present = [], []
    runs = [
        clips_and_pairs(items[j : j + micro_batch])
        for j in range(0, len(items), micro_batch)
    ]
    batches = (paths for paths, _ in runs)
    with closing(prefetch_clips(read, batches)) as prefetched:
        for micro_index, (paths, pairs) in enumerate(runs):
            with phase("data.input_wait"):
                clips = next(prefetched)
            # Compiled MLP backward may consume buffers needed by repeated gradients.
            with (
                torch.compiler.set_stance("force_eager")
                if selected is not None
                else nullcontext()
            ):
                loss, result = micro_loss(
                    modules, clips, pairs, factors, compile_tensors=selected is None
                )
                if selected is None:
                    with phase("backward"):
                        loss.backward()
                else:
                    groups, directory, progress = selected
                    with phase("backward"):
                        loss.backward(retain_graph=True)
                    work_records.append(
                        collect_micro(
                            result.weighted, modules, groups, directory, micro_index
                        )
                    )
                    if progress is not None:
                        progress.update()
                        progress.refresh()
            with phase("diagnostics.observations"):
                policy, state_rows = policy_observation(
                    result.policy, cfg.calibration_bins, cfg.action_copy_eps
                )
                micro.append(
                    MicroObservations(
                        preference_observation(result.preference, deferred=True),
                        policy,
                        phi_observation(result.phi),
                        state_rows,
                        raw_policy_observation(
                            result.policy,
                            clips,
                            getattr(modules["pi_loss"], "masks", None),
                        ),
                    )
                )
                scalar_rows.append(logging_row(result, loss))
                p_present.append(result.components["P"] is not None)
                sum_p += sum(pair[2]["p"] for pair in pairs if pair[2]["inc"] == 0)
                n_k, n_r = n_k + len(paths), n_r + len(pairs)
            del clips, loss, result
    with phase("diagnostics.reduce"):
        totals, component_totals, dual_lambda = finish_scalars(
            scalar_rows, p_present, modules["phi_loss"].log_lambda.exp().detach()
        )
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
        "dual_lambda": dual_lambda,
    }
    return logged, work_records


def step_gradients(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_phi: float,
    w_pi: float,
    w_P: float,
    micro_batch: int,
    *,
    cfg: PretrainConfig,
) -> dict:
    """Original ordered backwards, followed by batched CPU observation transfers."""
    return gradient_steps(
        items, modules, read, m, k_tot, r_tot, w_phi, w_pi, w_P, micro_batch, cfg, None
    )[0]


def selected_step_gradients(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    read: Callable[[str], Clip],
    m: int,
    k_tot: int,
    r_tot: int,
    w_phi: float,
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
        w_phi,
        w_pi,
        w_P,
        micro_batch,
        cfg,
        (groups, directory, progress),
    )
