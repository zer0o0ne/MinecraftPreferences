"""Original micro-batch scalar and exact weighted components of (23)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from mcpref.datasets.clips.io import Clip
from mcpref.infrastructure.profiling.scopes import measured, phase

Pairs = list[tuple[int, int, dict]]
"""R(B): per labeled item in order, i_a, i_b and its labels."""


@dataclass(frozen=True)
class MicroResult:
    sums: list[Tensor]
    components: dict
    preference: tuple[Tensor, Tensor, list[dict]] | None
    policy: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]
    phi: dict[str, Tensor]
    weighted: dict[str, Tensor | None]


def component_sums(terms: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().sum() for key, value in terms.items() if key != "total"}


@measured("micro.forward")
def micro_loss(
    modules: dict[str, torch.nn.Module],
    clips: list[Clip],
    pairs: Pairs,
    factors: tuple[float, float, float, float],
    compile_tensors: bool = False,
) -> tuple[torch.Tensor, MicroResult]:
    """The original scalar, exact weighted components and detached observations."""
    k_factor, w_phi, w_pi, r_factor = factors
    states = modules["phi"].prepare(clips)
    Z = modules["phi"](clips, states)
    compiled = compile_tensors and Z.is_cuda
    phi_loss = (
        modules["phi_loss"].compiled_terms if compiled else modules["phi_loss"].terms
    )
    with phase("heads_losses"):
        phi_terms, phi_diagnostics = phi_loss(Z, clips, states)
        pi_out = modules["pi"](Z, clips)
        pi_terms, valid, policy_diagnostics = modules["pi_loss"].terms(pi_out, Z, clips)
    q = policy_diagnostics["q"]
    current = policy_diagnostics["current"]
    zero_shift = policy_diagnostics["zero_shift"]
    l_phi, l_pi = phi_terms["total"], pi_terms["total"]
    loss = k_factor * (w_phi * l_phi + w_pi * l_pi).sum()
    weighted = {
        "phi": k_factor * w_phi * l_phi.sum(),
        "pi": k_factor * w_pi * l_pi.sum(),
        "P": None,
    }
    sums = [l_phi.detach().sum(), l_pi.detach().sum(), l_phi.detach().new_zeros(())]
    components = {
        "phi": component_sums(phi_terms),
        "pi": component_sums(pi_terms),
        "P": None,
    }
    preference = None
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
            weighted["P"] = r_factor * l_P.sum()
            sums[2] = l_P.detach().sum()
            components["P"] = component_sums(preference_terms)
            preference = (
                preference_out[0].detach(),
                preference_out[1].detach(),
                labels,
            )
    policy = (
        pi_out[0].detach(),
        q,
        current,
        {key: value.detach() for key, value in valid.items()},
        zero_shift.detach(),
    )
    phi = {key: value.detach() for key, value in phi_diagnostics.items()}
    return loss, MicroResult(sums, components, preference, policy, phi, weighted)
