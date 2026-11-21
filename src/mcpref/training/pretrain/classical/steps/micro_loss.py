"""Clip-only auxiliary losses and pair-only preference loss of specs/pretrain.md (1)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from mcpref.datasets.clips.io import Clip
from mcpref.infrastructure.profiling.scopes import measured, phase

Pairs = list[tuple[int, int, dict]]
"""R(B): per labeled item in order, i_a, i_b and its labels."""


@dataclass
class MicroResult:
    sums: list[Tensor]
    components: dict
    preference: tuple[Tensor, Tensor, list[dict]] | None
    policy: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor] | None
    phi: dict[str, Tensor] | None
    weighted: dict[str, Tensor | None]
    clip_indices: list[int]


def component_sums(terms: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().sum() for key, value in terms.items() if key != "total"}


def clip_loss(modules, Z, clips, states, factors, compiled, result) -> Tensor:
    """Compute own losses only for standalone clip rows; preserve their autograd paths."""
    k_factor, w_phi, w_pi, _ = factors
    phi_loss = (
        modules["phi_loss"].compiled_terms if compiled else modules["phi_loss"].terms
    )
    with phase("heads_losses"):
        phi_terms, phi_diagnostics = phi_loss(Z, clips, states)
        pi_out = modules["pi"](Z, clips)
        pi_terms, valid, diagnostics = modules["pi_loss"].terms(pi_out, Z, clips)
    l_phi, l_pi = phi_terms["total"], pi_terms["total"]
    result.weighted["phi"] = k_factor * w_phi * l_phi.sum()
    result.weighted["pi"] = k_factor * w_pi * l_pi.sum()
    result.sums[:2] = [l_phi.detach().sum(), l_pi.detach().sum()]
    result.components["phi"] = component_sums(phi_terms)
    result.components["pi"] = component_sums(pi_terms)
    result.policy = (
        pi_out[0].detach(),
        diagnostics["q"].detach(),
        diagnostics["current"].detach(),
        {key: value.detach() for key, value in valid.items()},
        diagnostics["zero_shift"].detach(),
    )
    result.phi = {key: value.detach() for key, value in phi_diagnostics.items()}
    return k_factor * (w_phi * l_phi + w_pi * l_pi).sum()


@measured("micro.forward")
def micro_loss(
    modules: dict[str, torch.nn.Module],
    clips: list[Clip],
    pairs: Pairs,
    factors: tuple[float, float, float, float],
    compile_tensors: bool = False,
) -> tuple[torch.Tensor, MicroResult]:
    """Encode every required history; route each item to its own objective."""
    r_factor = factors[3]
    states = modules["phi"].prepare(clips)
    Z = modules["phi"](clips, states)
    compiled = compile_tensors and Z.is_cuda
    endpoints = {i for a, b, _ in pairs for i in (a, b)}
    indices = [i for i in range(len(clips)) if i not in endpoints]
    zero = Z.new_zeros(())
    result = MicroResult(
        [zero, zero, zero],
        dict.fromkeys(("phi", "pi", "P")),
        None,
        None,
        None,
        dict.fromkeys(("phi", "pi", "P")),
        indices,
    )
    loss = zero
    if indices:
        loss = clip_loss(
            modules,
            Z[indices],
            [clips[i] for i in indices],
            [states[i] for i in indices],
            factors,
            compiled,
            result,
        )
    with phase("preference_loss"):
        if pairs:
            i_a, i_b, labels = (list(x) for x in zip(*pairs))
            preference_loss = (
                modules["P_loss"].compiled_terms
                if compiled
                else modules["P_loss"].terms
            )
            preference_out = modules["P"](Z[i_a], Z[i_b])
            preference_terms = preference_loss(preference_out, labels)
            l_P = preference_terms["total"]
            loss = loss + r_factor * l_P.sum()
            result.weighted["P"] = r_factor * l_P.sum()
            result.sums[2] = l_P.detach().sum()
            result.components["P"] = component_sums(preference_terms)
            result.preference = (
                preference_out[0].detach(),
                preference_out[1].detach(),
                labels,
            )
    return loss, result
