"""Fixed-reference evaluation of specs/pretrain-diagnostics.md (18)-(21)."""

import copy
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from tqdm import tqdm

from mcpref import REPO
from mcpref.pretrain.checkpoints import Modules, cpu_states
from mcpref.pretrain.clip import Clip, read_clip
from mcpref.pretrain.config import PretrainConfig
from mcpref.pretrain.diagnostics import GlobalFit
from mcpref.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
)
from mcpref.pretrain.reference import PARTS, ReferenceData
from mcpref.pretrain.reference_metrics import (
    add_preference,
    empty_part,
    finish_part,
    preference_data,
    shared_initial,
)

CONDITIONS = ("joint", "P_only", "phi_only")


@contextmanager
def evaluation_state(cfg: PretrainConfig, modules: Modules) -> Iterator[None]:
    """(19): eval/no_grad with configured-device RNG and mode restoration."""
    device = torch.device(cfg.device)
    modes = {name: module.training for name, module in modules.items()}
    buffers = {
        name: {
            buffer_name: buffer.detach().clone()
            for buffer_name, buffer in module.named_buffers()
        }
        for name, module in modules.items()
    }
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        for module in modules.values():
            module.eval()
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for name, module in modules.items():
                for buffer_name, buffer in module.named_buffers():
                    buffer.copy_(buffers[name][buffer_name])
        for name, module in modules.items():
            module.train(modes[name])
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)


def clip_reader(run: dict):
    def read(path: str) -> Clip:
        return read_clip(REPO / path, run["states"], run["width"], run["height"])

    return read


def observe_joint(
    cfg: PretrainConfig,
    modules: Modules,
    Z: torch.Tensor,
    clips: list[Clip],
    joint: tuple,
    part: dict,
) -> None:
    """Evaluate current phi/pi once and retain bounded detached joint observations."""
    phi_terms, phi_diagnostics = modules["phi_loss"].terms(Z, clips)
    pi_out = modules["pi"](Z, clips)
    pi_terms, valid = modules["pi_loss"].terms(pi_out, Z, clips)
    q, current = modules["pi_loss"].diagnostic_targets()
    zero_shift = modules["pi_loss"].zero_shift(Z, valid["shift_valid"])
    policy, state_rows = policy_observation(
        (pi_out[0].detach(), q, current, valid, zero_shift.detach()),
        cfg.calibration_bins,
        cfg.action_copy_eps,
    )
    part["micro"].append(
        MicroObservations(
            joint[1], policy, phi_observation(phi_diagnostics), state_rows
        )
    )
    for key, value in phi_terms.items():
        part["phi"][key].extend(value.detach().cpu().tolist())
    for key, value in pi_terms.items():
        part["pi"][key].extend(value.detach().cpu().tolist())


def evaluate_pair(
    cfg: PretrainConfig,
    modules: Modules,
    item: dict,
    j: int,
    read,
    reference: ReferenceData | None,
    frozen: torch.nn.Module | None,
    latents: list[torch.Tensor],
    parts: dict,
) -> None:
    """Evaluate all due conditions for one selected pair in the specified call order."""
    clips = [read(item["a"]), read(item["b"])]
    Z = modules["phi"](clips)
    if reference is None:
        latents.append(Z.detach().to("cpu", copy=True))
    joint = preference_data(modules, modules["P"](Z[:1], Z[1:]), item["labels"])
    part = parts[item["part"]]
    add_preference(part, "joint", joint)
    if reference is not None:
        z0 = reference.Z0[j].to(torch.device(cfg.device))
        only_P = modules["P"](z0[0:1], z0[1:2])
        add_preference(part, "P_only", preference_data(modules, only_P, item["labels"]))
        add_preference(
            part,
            "phi_only",
            preference_data(modules, frozen(Z[:1], Z[1:]), item["labels"]),
        )
    observe_joint(cfg, modules, Z, clips, joint, part)


def evaluate_conditions(
    cfg: PretrainConfig,
    modules: Modules,
    selection: dict,
    baselines: GlobalFit,
    run: dict,
    reference: ReferenceData | None,
) -> tuple[dict, list[torch.Tensor]]:
    """Evaluate each selected pair once, adding the other conditions at positive steps."""
    initial = reference is None
    conditions = CONDITIONS[:1] if initial else CONDITIONS
    parts = {name: empty_part(conditions) for name in PARTS}
    latents, read = [], clip_reader(run)
    desc = "pretrain reference initial" if initial else "pretrain reference"
    with evaluation_state(cfg, modules):
        frozen = None
        if reference is not None:
            frozen = copy.deepcopy(modules["P"])
            frozen.load_state_dict(reference.P0, strict=True)
            frozen.eval()
        with tqdm(
            total=len(selection["items"]),
            desc=desc,
            unit="pair",
            file=sys.stderr,
            mininterval=0,
        ) as bar:
            bar.refresh()
            for j, item in enumerate(selection["items"]):
                evaluate_pair(
                    cfg, modules, item, j, read, reference, frozen, latents, parts
                )
                bar.update()
                bar.refresh()
        metrics = {
            name: finish_part(cfg, modules, baselines, part, conditions)
            for name, part in parts.items()
        }
        if initial:
            conditions_by_name = shared_initial(metrics, CONDITIONS)
        else:
            conditions_by_name = {
                condition: {part: metrics[part][condition] for part in PARTS}
                for condition in CONDITIONS
            }
            conditions_by_name["initial"] = copy.deepcopy(
                reference.initial_metrics["conditions"]["initial"]
            )
    return conditions_by_name, latents


def evaluate_initial(
    cfg: PretrainConfig,
    modules: Modules,
    selection: dict,
    baselines: GlobalFit,
    run: dict,
) -> tuple[dict[str, torch.Tensor | dict[str, torch.Tensor]], dict]:
    """(18): step-0 latents/state and four shared preference conditions."""
    conditions, latents = evaluate_conditions(
        cfg, modules, selection, baselines, run, None
    )
    initial = {
        "Z0": torch.stack(latents),
        "P0": cpu_states({"P": modules["P"]})["P"],
    }
    metrics = {"version": 1, "step": 0, "conditions": conditions}
    return initial, metrics


def evaluate_reference(
    cfg: PretrainConfig,
    modules: Modules,
    reference: ReferenceData,
    run: dict,
    step: int,
) -> dict:
    """(20)-(21): current/frozen preference conditions at a positive due step."""
    conditions, _ = evaluate_conditions(
        cfg, modules, reference.selection, reference.baselines, run, reference
    )
    return {"version": 1, "step": step, "conditions": conditions}
