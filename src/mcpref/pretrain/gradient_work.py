"""Bounded step-summed gradient reduction of specs/pretrain-diagnostics.md (25)."""

import json
import math
from pathlib import Path

import torch
from torch import Tensor

from mcpref.pretrain.checks import Groups
from mcpref.pretrain.gradient_collect import (
    add_counts,
    collect_micro,
    nonfinite,
    zero_counts,
)
from mcpref.shards.store import complete_shards, shard_path

__all__ = ["collect_micro"]


def work_manifests(directory: Path) -> list[dict]:
    return [
        json.loads((shard_path(directory, index) / "manifest.json").read_bytes())
        for index in range(complete_shards(directory))
    ]


def contribution_count(manifests: list[dict], component: str | None = None) -> int:
    return sum(
        not entry["zero"] and (component is None or entry["component"] == component)
        for manifest in manifests
        for entry in manifest.values()
    )


def load_vector(
    directory: Path,
    manifests: list[dict],
    x: str,
    component: str,
    parameter: str,
    progress=None,
    counts: dict | None = None,
) -> Tensor:
    total = None
    shape, dtype = None, None
    for index, manifest in enumerate(manifests):
        for identifier, entry in manifest.items():
            if (entry["x"], entry["component"], entry["parameter"]) != (
                x,
                component,
                parameter,
            ):
                continue
            shape, dtype = entry["shape"], getattr(torch, entry["dtype"])
            if entry["zero"]:
                continue
            value = torch.load(
                shard_path(directory, index) / "tensors" / f"{identifier}.pt",
                map_location="cpu",
                weights_only=True,
            )
            total = value.clone() if total is None else total + value
            if counts is not None:
                add_counts(counts[x][component][entry["group"]], nonfinite(value))
            if progress is not None:
                progress.update()
                progress.refresh()
    if total is None:
        total = torch.zeros(shape, dtype=dtype)
    return total


def squared(value: Tensor) -> float:
    return torch.linalg.vector_norm(value, ord=2).item() ** 2


def dot(a: Tensor, b: Tensor) -> float:
    return (a.to(torch.float64) * b.to(torch.float64)).sum().item()


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def phi_bundle(raw: dict) -> dict:
    norms = {key: finite(math.sqrt(raw["norms"][key])) for key in ("phi", "pi", "P")}
    dots = {
        key: finite(raw["dot"][key]) for key in ("P_phi", "P_pi", "P_other", "P_joint")
    }
    cosines = {}
    for key, other in (("P_phi", "phi"), ("P_pi", "pi"), ("P_other", "other")):
        denominator = math.sqrt(raw["norms"]["P"] * raw["norms"][other])
        cosines[key] = finite(raw["dot"][key] / denominator) if denominator else None
    return {
        "norms": norms,
        "dot": dots,
        "cosine": cosines,
        "update_dot": None,
        "update_unavailable": None,
    }


def raw_phi() -> dict:
    return {
        "norms": dict.fromkeys(("phi", "pi", "P", "other"), 0.0),
        "dot": dict.fromkeys(("P_phi", "P_pi", "P_other", "P_joint"), 0.0),
    }


def add_raw(target: dict, source: dict) -> None:
    for key, metrics in target.items():
        for metric in metrics:
            metrics[metric] += source[key][metric]


def reduce_work_with_progress(directory: Path, groups: Groups, progress) -> dict:
    """(25): reduce in Groups order, with progress kept outside the public interface."""
    manifests = work_manifests(directory)
    raw = {"phi": {}, "P": {}}
    counts = {"phi": {"phi": {}, "pi": {}, "P": {}}, "P": {"P": {}}}
    present = {
        x: {
            component: any(
                entry["x"] == x and entry["component"] == component
                for manifest in manifests
                for entry in manifest.values()
            )
            for component in (("phi", "pi", "P") if x == "phi" else ("P",))
        }
        for x in ("phi", "P")
    }
    for group, parameters in groups.items():
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            parameter_name = groups.parameter_names[id(parameter)]
            if parameter_name.startswith("P."):
                x = "P"
            elif parameter_name.startswith(("phi.", "phi_loss.")):
                x = "phi"
            else:
                continue
            components = ("phi", "pi", "P") if x == "phi" else ("P",)
            for component in components:
                counts[x][component].setdefault(group, zero_counts())
            vectors = {
                component: load_vector(
                    directory, manifests, x, component, parameter_name, progress, counts
                )
                for component in components
            }
            if x == "phi":
                values = raw["phi"].setdefault(group, raw_phi())
                for component in ("phi", "pi", "P"):
                    values["norms"][component] += squared(vectors[component])
                values["norms"]["other"] += squared(vectors["phi"] + vectors["pi"])
                values["dot"]["P_phi"] += dot(vectors["P"], vectors["phi"])
                values["dot"]["P_pi"] += dot(vectors["P"], vectors["pi"])
                values["dot"]["P_other"] += dot(
                    vectors["P"], vectors["phi"] + vectors["pi"]
                )
                values["dot"]["P_joint"] += dot(
                    vectors["P"], vectors["phi"] + vectors["pi"] + vectors["P"]
                )
            else:
                values = raw["P"].setdefault(group, 0.0)
                raw["P"][group] = values + squared(vectors["P"])
    phi_total = raw_phi()
    for values in raw["phi"].values():
        add_raw(phi_total, values)
    result = {
        "phi": {
            "groups": {
                group: phi_bundle(values) for group, values in raw["phi"].items()
            },
            "total": phi_bundle(phi_total),
        },
        "P": {
            "groups": {
                group: {
                    "norm": finite(math.sqrt(value)),
                    "update_dot": None,
                    "update_unavailable": None,
                }
                for group, value in raw["P"].items()
            },
            "total": {
                "norm": finite(math.sqrt(sum(raw["P"].values()))),
                "update_dot": None,
                "update_unavailable": None,
            },
        },
    }
    result["nonfinite"] = {
        x: {
            component: counts[x][component] if present[x][component] else None
            for component in counts[x]
        }
        for x in counts
    }
    return result


def reduce_work(directory: Path, *, groups: Groups) -> dict:
    """Interface: bounded reduction in optimizer Groups declaration order."""
    return reduce_work_with_progress(directory, groups, None)
