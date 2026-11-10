"""Fixed-reference condition evaluation and producer observations."""

import copy
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from tqdm import tqdm

from mcpref.diagnostics.core.types import ReferencePair
from mcpref.diagnostics.preference.suites import PreferenceRows, pool_preferences
from mcpref.diagnostics.pretrain.diagnostics import GlobalFit
from mcpref.training.checkpoints import Modules, cpu_states
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.references.metrics import (
    empty_part,
    finish_part,
    shared_initial,
)
from mcpref.training.pretrain.references.pairs import (
    InitialPairs,
    clip_reader,
    evaluate_pair,
)
from mcpref.training.pretrain.references.reference import PARTS, ReferenceData

CONDITIONS = ("joint", "P_only", "phi_only")
Observations = dict[str, dict[str, PreferenceRows | None]]


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


def evaluate_conditions(
    cfg: PretrainConfig,
    modules: Modules,
    selection: dict,
    baselines: GlobalFit,
    run: dict,
    reference: ReferenceData | None,
    initial_pairs: InitialPairs = None,
) -> tuple[dict, list[torch.Tensor], list[ReferencePair], Observations]:
    """Evaluate each selected pair once, adding the other conditions at positive steps."""
    initial = reference is None
    conditions = CONDITIONS[:1] if initial else CONDITIONS
    parts = {name: empty_part(conditions) for name in PARTS}
    latents, read, pair_records = [], clip_reader(run), []
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
                pair = evaluate_pair(
                    cfg,
                    modules,
                    item,
                    j,
                    read,
                    reference,
                    frozen,
                    latents,
                    parts,
                    initial_pairs,
                )
                if pair is not None:
                    pair_records.append(pair)
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
    observations = {
        condition: {
            part: pool_preferences(
                parts[part]["conditions"]["joint" if initial else condition]["rows"]
            )
            for part in PARTS
        }
        for condition in CONDITIONS
    }
    observations["initial"] = (
        copy.deepcopy(observations["joint"]) if initial else dict.fromkeys(PARTS)
    )
    return conditions_by_name, latents, pair_records, observations


def evaluate_initial_event(
    cfg: PretrainConfig,
    modules: Modules,
    selection: dict,
    baselines: GlobalFit,
    run: dict,
) -> tuple[
    dict[str, torch.Tensor | dict[str, torch.Tensor]],
    dict,
    list[ReferencePair],
    Observations,
]:
    conditions, latents, pairs, observations = evaluate_conditions(
        cfg, modules, selection, baselines, run, None
    )
    initial = {
        "Z0": torch.stack(latents),
        "P0": cpu_states({"P": modules["P"]})["P"],
    }
    metrics = {"version": 1, "step": 0, "conditions": conditions}
    return initial, metrics, pairs, observations


def evaluate_reference_event(
    cfg: PretrainConfig,
    modules: Modules,
    reference: ReferenceData,
    run: dict,
    step: int,
) -> tuple[dict, list[ReferencePair] | None, Observations]:
    conditions, _, pairs, observations = evaluate_conditions(
        cfg,
        modules,
        reference.selection,
        reference.baselines,
        run,
        reference,
        reference.initial_pairs,
    )
    metrics = {"version": 1, "step": step, "conditions": conditions}
    return metrics, pairs if reference.initial_pairs is not None else None, observations
