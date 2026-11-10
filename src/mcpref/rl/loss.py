"""Eqs. (2)-(3): batch-1 RL losses and globally normalised micro-batch backward."""

from collections.abc import Callable

import torch
from torch import Tensor

from mcpref import REPO
from mcpref.diagnostics.observations import PpoObservations, ShiftObservations
from mcpref.perception.config import PerceptionConfig
from mcpref.perception.loss import latent_metric
from mcpref.ppo.config import PpoConfig
from mcpref.ppo.loss import ppo_loss, shift_loss
from mcpref.ppo.reward import reward
from mcpref.rl.history import encode_ref
from mcpref.rl_pairs.types import StateRef
from mcpref.sampling.config import SamplingConfig
from mcpref.sampling.shards import load_step

ROLES = ("ppo", "shift", "metra", "P")


def state_ref(sampling: SamplingConfig, index: int) -> StateRef:
    """The owning trajectory's episode identity of one global action/state index."""
    cycle = index // sampling.steps
    return StateRef(f"{sampling.out_dir}#{cycle:06d}", index)


def ppo_eligible(item: dict, sampling: SamplingConfig, ppo: PpoConfig) -> bool:
    kind, t = item["kind"], item["index"]
    return (
        kind == "pending"
        or t + ppo.horizon <= (t // sampling.steps + 1) * sampling.steps
    )


def micro_loss(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    sampling: SamplingConfig,
    perception: PerceptionConfig,
    states: int,
    alpha: Tensor,
    ppo: PpoConfig,
    discount: float,
    min_p: float,
    metric: Callable[[Tensor], Tensor],
    observations: dict | None = None,
    fixed_advantages: dict[tuple[str, int], float] | None = None,
) -> tuple[dict[str, list[Tensor]], dict[str, list[Tensor]]]:
    """Compute each item's supplied roles with one shared fresh policy readout."""
    phi, pi, P = modules["phi"], modules["pi"], modules["P"]
    terms = {role: [] for role in ROLES}
    logged = {role: [] for role in ROLES}
    action_count = len(pi.names)

    def encode(ref: StateRef):
        return encode_ref(ref, phi, perception, sampling, states, action_count)

    for item in items:
        kind, t = item["kind"], item.get("index")
        if kind == "preference":
            a = encode(StateRef(**item["a"]))
            b = encode(StateRef(**item["b"]))
            value = modules["P_loss"](P(a.Y, b.Y), [item["labels"]])[0]
            terms["P"].append(value)
            logged["P"].append(value.detach())
            continue
        origin = encode(state_ref(sampling, t))
        record = load_step(REPO / sampling.out_dir, t)
        R = pi.readout(origin.Y, origin.C)[:, -1]
        eligible = ppo_eligible(item, sampling, ppo)
        successor = encode(state_ref(sampling, t + 1)) if kind == "fresh" else None
        if eligible:
            logits = pi.logits(R)
            if fixed_advantages is None:
                with torch.no_grad():
                    futures = [
                        encode(state_ref(sampling, t + k)).Y
                        for k in range(1 if successor is None else 2, ppo.horizon + 1)
                    ]
                if successor is not None:
                    futures = [successor.Y] + futures
                advantage = reward(
                    P, [origin.Y], [futures], ppo.horizon, discount, min_p
                )
            else:
                advantage = torch.tensor(
                    [fixed_advantages[kind, t]],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            actions = torch.tensor(
                [record.action], dtype=torch.int64, device=logits.device
            )
            behavior = (
                record.probabilities[record.action]
                .reshape(1)
                .to(device=logits.device, dtype=logits.dtype)
            )
            value = ppo_loss(logits, actions, behavior, advantage, ppo.clip)[0]
            terms["ppo"].append(value)
            logged["ppo"].append(value.detach())
            if observations is not None:
                probability = torch.softmax(logits, dim=-1, dtype=None)
                ratio = probability[0, record.action] / behavior[0]
                observations["ppo"].append(
                    PpoObservations(
                        ratio.detach().reshape(1),
                        advantage.detach().reshape(1),
                        record.probabilities.reshape(1, -1).to(
                            device=logits.device, dtype=logits.dtype
                        ),
                        probability.detach(),
                        actions,
                        ppo.clip,
                        (kind,),
                        torch.tensor([True], device=logits.device),
                    )
                )
                observations["ppo_clusters"].append(state_ref(sampling, t).episode)
                observations["logits"].append(logits.detach())
        if kind == "pending":
            continue
        actions = torch.tensor([record.action], dtype=torch.int64, device=R.device)
        predicted = pi.shift(R, actions)
        actual = successor.Y[0, -1:] - origin.Y[0, -1:]
        shift = shift_loss(predicted, actual, metric)[0]
        metra = modules["metra"](actual, predicted, alpha)[0]
        terms["shift"].append(shift)
        terms["metra"].append(metra)
        logged["shift"].append(shift.detach())
        logged["metra"].append(metra.detach())
        if observations is not None:
            distance = metric(actual).detach()
            observations["shift"].append(
                ShiftObservations(
                    predicted.detach(),
                    actual.detach(),
                    alpha,
                    distance,
                    torch.clamp(1 - distance**2, max=modules["metra"].cfg.dual_slack),
                    modules["metra"].log_lambda.exp().detach().reshape(1),
                    modules["metra"].cfg.dual_slack,
                    (
                        torch.linalg.vector_norm(
                            predicted.detach() * alpha, ord=2, dim=-1
                        )
                        == 0
                    ),
                )
            )
            observations["shift_clusters"].append(state_ref(sampling, t).episode)
    return terms, logged


def step_gradients(
    items: list[dict],
    modules: dict[str, torch.nn.Module],
    *,
    sampling: SamplingConfig,
    perception: PerceptionConfig,
    states: int,
    alpha: Tensor,
    ppo: PpoConfig,
    discount: float,
    min_p: float,
    m: int,
    n_a: int,
    n_delta: int,
    r_c: int,
    w_ppo: float,
    w_delta: float,
    w_metra: float,
    w_P: float,
    micro_batch: int,
    observations: dict | None = None,
    fixed_advantages: dict[tuple[str, int], float] | None = None,
) -> dict:
    """Accumulate Eq. (3) into existing gradients, one backward per micro-batch."""
    metric = latent_metric(perception)
    logged = {role: [] for role in ROLES}
    totals = []
    for start in range(0, len(items), micro_batch):
        terms, values = micro_loss(
            items[start : start + micro_batch],
            modules,
            sampling,
            perception,
            states,
            alpha,
            ppo,
            discount,
            min_p,
            metric,
            observations,
            fixed_advantages,
        )
        total = None
        if n_a and terms["ppo"]:
            total = m * w_ppo / n_a * sum(terms["ppo"])
        if terms["shift"]:
            transition = sum(
                w_delta * shift + w_metra * metra
                for shift, metra in zip(terms["shift"], terms["metra"])
            )
            part = m / n_delta * transition
            total = part if total is None else total + part
        if r_c and terms["P"]:
            part = m * w_P / r_c * sum(terms["P"])
            total = part if total is None else total + part
        if total is None:
            raise RuntimeError("a non-empty micro-batch supplies no loss")
        total.backward()
        totals.append(total.detach())
        for role in ROLES:
            logged[role].extend(values[role])
    scalars = (
        torch.stack([*totals, *(value for role in ROLES for value in logged[role])])
        .cpu()
        .tolist()
    )
    total_value = 0.0
    for value in scalars[: len(totals)]:
        total_value += value
    result, offset = {}, len(totals)
    for role in ROLES:
        values = scalars[offset : offset + len(logged[role])]
        offset += len(logged[role])
        result[role] = sum(values) / len(values) if values else None
    return result | {"loss": total_value}
