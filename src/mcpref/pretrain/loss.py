"""(1), (2), (7) of specs/pretrain.md: the gradients of a mini-batch's loss accumulated
over its micro-batches, and the logged losses."""

from collections.abc import Callable

import torch

from mcpref.pretrain.clip import Clip

Pairs = list[tuple[int, int, dict]]
"""R(B): per labeled item in order, i_a, i_b and its labels."""


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


def micro_loss(
    modules: dict[str, torch.nn.Module],
    clips: list[Clip],
    pairs: Pairs,
    factors: tuple[float, float, float],
) -> tuple[torch.Tensor, list[float]]:
    """(2): L_j, the terms of (1) with factors M / K_tot, w_pi, M w_P / R_tot, the calls
    in the order phi, l^phi, pi, l^pi, P, l^P; the sums of l^phi, l^pi, l^P for (7)."""
    k_factor, w_pi, r_factor = factors
    Z = modules["phi"](clips)
    l_phi = modules["phi_loss"](Z, clips)
    l_pi = modules["pi_loss"](modules["pi"](Z), Z, clips)
    loss = k_factor * (l_phi + w_pi * l_pi).sum()
    sums = [l_phi.detach().sum().item(), l_pi.detach().sum().item(), 0.0]
    if pairs:
        i_a, i_b, labels = (list(x) for x in zip(*pairs))
        l_P = modules["P_loss"](modules["P"](Z[i_a], Z[i_b]), labels)
        loss = loss + r_factor * l_P.sum()
        sums[2] = l_P.detach().sum().item()
    return loss, sums


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
) -> dict:
    """(2) on the mini-batch `items`: each L_j.backward() in order, accumulating into
    .grad; (7) without lr: {phi, pi, P (None without pairs), loss}."""
    factors = (m / k_tot, w_pi, m * w_P / r_tot if r_tot else 0.0)
    totals, n_k, n_r = [0.0] * 4, 0, 0
    for j in range(0, len(items), micro_batch):
        paths, pairs = clips_and_pairs(items[j : j + micro_batch])
        loss, sums = micro_loss(modules, [read(p) for p in paths], pairs, factors)
        loss.backward()
        totals = [a + b for a, b in zip(totals, [*sums, loss.detach().item()])]
        n_k, n_r = n_k + len(paths), n_r + len(pairs)
    phi, pi, P, loss = totals
    return {
        "phi": phi / n_k,
        "pi": pi / n_k,
        "P": P / n_r if n_r else None,
        "loss": loss,
    }
