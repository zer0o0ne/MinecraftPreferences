"""Original micro-batch scalar and exact weighted components of (23)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from mcpref.pretrain.clip import Clip

Pairs = list[tuple[int, int, dict]]
"""R(B): per labeled item in order, i_a, i_b and its labels."""


@dataclass(frozen=True)
class MicroResult:
    sums: list[float]
    components: dict
    preference: tuple[Tensor, Tensor, list[dict]] | None
    policy: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]
    phi: dict[str, Tensor]
    weighted: dict[str, Tensor | None]


def component_sums(terms: dict[str, Tensor]) -> dict[str, float]:
    return {
        key: value.detach().sum().item()
        for key, value in terms.items()
        if key != "total"
    }


def micro_loss(
    modules: dict[str, torch.nn.Module],
    clips: list[Clip],
    pairs: Pairs,
    factors: tuple[float, float, float],
) -> tuple[torch.Tensor, MicroResult]:
    """The original scalar, exact weighted components and detached observations."""
    k_factor, w_pi, r_factor = factors
    Z = modules["phi"](clips)
    phi_terms, phi_diagnostics = modules["phi_loss"].terms(Z, clips)
    pi_out = modules["pi"](Z, clips)
    pi_terms, valid = modules["pi_loss"].terms(pi_out, Z, clips)
    q, current = modules["pi_loss"].diagnostic_targets()
    zero_shift = modules["pi_loss"].zero_shift(Z, valid["shift_valid"])
    l_phi, l_pi = phi_terms["total"], pi_terms["total"]
    loss = k_factor * (l_phi + w_pi * l_pi).sum()
    weighted = {
        "phi": k_factor * l_phi.sum(),
        "pi": k_factor * w_pi * l_pi.sum(),
        "P": None,
    }
    sums = [l_phi.detach().sum().item(), l_pi.detach().sum().item(), 0.0]
    components = {
        "phi": component_sums(phi_terms),
        "pi": component_sums(pi_terms),
        "P": None,
    }
    preference = None
    if pairs:
        i_a, i_b, labels = (list(x) for x in zip(*pairs))
        preference_out = modules["P"](Z[i_a], Z[i_b])
        preference_terms = modules["P_loss"].terms(preference_out, labels)
        l_P = preference_terms["total"]
        loss = loss + r_factor * l_P.sum()
        weighted["P"] = r_factor * l_P.sum()
        sums[2] = l_P.detach().sum().item()
        components["P"] = component_sums(preference_terms)
        preference = (preference_out[0].detach(), preference_out[1].detach(), labels)
    policy = (
        pi_out[0].detach(),
        q,
        current,
        {key: value.detach() for key, value in valid.items()},
        zero_shift.detach(),
    )
    phi = {key: value.detach() for key, value in phi_diagnostics.items()}
    return loss, MicroResult(sums, components, preference, policy, phi, weighted)
