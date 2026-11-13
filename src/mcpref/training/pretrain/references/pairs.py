"""Per-pair joint, P-only and phi-only fixed-reference evaluation."""

import torch

from mcpref.datasets.clips.io import Clip
from mcpref.diagnostics.core.types import ReferencePair
from mcpref.diagnostics.pretrain.observation import (
    MicroObservations,
    phi_observation,
    policy_observation,
)
from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import PretrainConfig
from mcpref.training.pretrain.references.metrics import add_preference, preference_data
from mcpref.training.pretrain.references.reference import ReferenceData

InitialPairs = list[ReferencePair] | tuple[ReferencePair, ...] | None


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
    pi_terms, valid, diagnostics = modules["pi_loss"].terms(pi_out, Z, clips)
    policy, state_rows = policy_observation(
        (
            pi_out[0].detach(),
            diagnostics["q"],
            diagnostics["current"],
            valid,
            diagnostics["zero_shift"],
        ),
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
    initial_pairs: InitialPairs,
) -> ReferencePair | None:
    """Evaluate all due conditions for one selected pair in the specified call order."""
    clips = [read(item["a"]), read(item["b"])]
    Z = modules["phi"](clips)
    if reference is None:
        latents.append(Z.detach().to("cpu", copy=True))
    joint = preference_data(modules, modules["P"](Z[:1], Z[1:]), item["labels"])
    part = parts[item["part"]]
    add_preference(part, "joint", joint)
    only_phi = joint
    if reference is not None:
        z0 = reference.Z0[j].to(torch.device(cfg.device))
        only_P = modules["P"](z0[0:1], z0[1:2])
        add_preference(part, "P_only", preference_data(modules, only_P, item["labels"]))
        only_phi = preference_data(modules, frozen(Z[:1], Z[1:]), item["labels"])
        add_preference(part, "phi_only", only_phi)
    observe_joint(cfg, modules, Z, clips, joint, part)
    if reference is not None and initial_pairs is None:
        return None
    baseline = joint[0]["total"] if reference is None else initial_pairs[j].baseline
    return ReferencePair(
        j,
        0,
        item["a"],
        item["b"],
        item["labels"],
        baseline,
        only_phi[0]["total"],
        joint[0]["total"],
        only_phi[0]["total"] - baseline,
        joint[0]["total"] - baseline,
    )
