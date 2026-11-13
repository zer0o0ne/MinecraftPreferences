"""Per-micro component gradient collection of specs/pretrain-diagnostics.md (24)."""

import math
from pathlib import Path

import torch
from torch import Tensor

from mcpref.infrastructure.profiling.scopes import measured
from mcpref.infrastructure.runs.files import json_bytes
from mcpref.infrastructure.shards.store import name, write_shard
from mcpref.training.checkpoints import Modules
from mcpref.training.groups import Groups


def training_device(modules: Modules) -> torch.device:
    for module in modules.values():
        for parameter in module.parameters():
            return parameter.device
    return torch.device("cpu")


def parameter_group(groups: Groups) -> dict[int, str]:
    return {
        id(parameter): group
        for group, parameters in groups.items()
        for parameter in parameters
    }


def parameters(
    modules: Modules, groups: Groups, x: str
) -> list[tuple[torch.nn.Parameter, str]]:
    names = ("phi", "phi_loss") if x == "phi" else ("P",)
    owner = parameter_group(groups)
    return [
        (parameter, owner[id(parameter)])
        for name in names
        for _, parameter in modules[name].named_parameters()
        if parameter.requires_grad
    ]


def zero_counts() -> dict:
    return {"nan": 0, "pos_inf": 0, "neg_inf": 0, "count": 0}


def nonfinite(value: Tensor) -> dict:
    nan = int(torch.isnan(value).sum().item())
    pos_inf = int(torch.isposinf(value).sum().item())
    neg_inf = int(torch.isneginf(value).sum().item())
    return {
        "nan": nan,
        "pos_inf": pos_inf,
        "neg_inf": neg_inf,
        "count": nan + pos_inf + neg_inf,
    }


def add_counts(target: dict, value: dict) -> None:
    for key in target:
        target[key] += value[key]


def micro_norms(params: dict, labelled: bool) -> dict:
    return {
        "phi": {
            group: {"phi": 0.0, "pi": 0.0, "P": 0.0 if labelled else None}
            for group in dict.fromkeys(group for _, group in params["phi"])
        },
        "P": (
            {
                group: {"P": 0.0}
                for group in dict.fromkeys(group for _, group in params["P"])
            }
            if labelled
            else None
        ),
    }


def micro_counts(norms: dict, labelled: bool) -> dict:
    return {
        "phi": {
            component: (
                {group: zero_counts() for group in norms["phi"]}
                if component != "P" or labelled
                else None
            )
            for component in ("phi", "pi", "P")
        },
        "P": {
            "P": ({group: zero_counts() for group in norms["P"]} if labelled else None)
        },
    }


def collect_gradients(
    weighted: dict,
    modules: Modules,
    params: dict,
    staged: Path,
    norms: dict,
    counts: dict,
    entries: dict,
    groups: Groups,
    labelled: bool,
) -> None:
    for x, components in (("phi", ("phi", "pi", "P")), ("P", ("P",))):
        for component in components:
            if component == "P" and not labelled:
                continue
            returned = torch.autograd.grad(
                weighted[component],
                [parameter for parameter, _ in params[x]],
                grad_outputs=torch.ones_like(weighted[component]),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
                materialize_grads=False,
            )
            for (parameter, group), gradient in zip(params[x], returned):
                identifier = name(len(entries))
                entries[identifier] = {
                    "x": x,
                    "component": component,
                    "group": group,
                    "parameter": groups.parameter_names[id(parameter)],
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype).removeprefix("torch."),
                    "zero": gradient is None,
                }
                if gradient is None:
                    continue
                value = gradient.detach().to("cpu", copy=True)
                torch.save(value, staged / "tensors" / f"{identifier}.pt")
                norms[x][group][component] += (
                    torch.linalg.vector_norm(value, ord=2, dtype=torch.float64).item()
                    ** 2
                )
                add_counts(counts[x][component][group], nonfinite(value))


@measured("diagnostics.component_gradients")
def collect_micro(
    weighted: dict[str, Tensor | None],
    modules: Modules,
    groups: Groups,
    directory: Path,
    micro_batch: int,
) -> dict:
    """(24): one autograd.grad call per x/component and one atomic CPU work shard."""
    device = training_device(modules)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        labelled = weighted["P"] is not None
        params = {x: parameters(modules, groups, x) for x in ("phi", "P")}
        norms = micro_norms(params, labelled)
        counts = micro_counts(norms, labelled)
        entries = {}
        with write_shard(directory, micro_batch) as staged:
            (staged / "tensors").mkdir()
            collect_gradients(
                weighted,
                modules,
                params,
                staged,
                norms,
                counts,
                entries,
                groups,
                labelled,
            )
            (staged / "manifest.json").write_bytes(json_bytes(entries))
        return {
            "micro_batch": micro_batch,
            "labelled": labelled,
            "losses": {
                key: value.detach().item() if value is not None else None
                for key, value in weighted.items()
            },
            "norms": {
                "phi": {
                    group: {
                        component: math.sqrt(value) if value is not None else None
                        for component, value in values.items()
                    }
                    for group, values in norms["phi"].items()
                },
                "P": (
                    {
                        group: {"P": math.sqrt(values["P"])}
                        for group, values in norms["P"].items()
                    }
                    if labelled
                    else None
                ),
            },
            "nonfinite": counts,
        }
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
