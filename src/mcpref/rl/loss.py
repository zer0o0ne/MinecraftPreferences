"""Eqs. (2)-(3): batch-1 RL losses and globally normalised micro-batch backward."""

import torch
from torch import Tensor

from mcpref import REPO
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
) -> tuple[dict[str, list[Tensor]], dict[str, list[float]]]:
    """Compute each item's supplied roles with one shared fresh policy readout."""
    phi, pi, P = modules["phi"], modules["pi"], modules["P"]
    metric = latent_metric(perception)
    terms = {role: [] for role in ROLES}
    logged = {role: [] for role in ROLES}
    action_count = len(pi.names)
    for item in items:
        kind, t = item["kind"], item.get("index")
        if kind == "preference":
            a = encode_ref(
                StateRef(**item["a"]), phi, perception, sampling, states, action_count
            )
            b = encode_ref(
                StateRef(**item["b"]), phi, perception, sampling, states, action_count
            )
            value = modules["P_loss"](P(a.Y, b.Y), [item["labels"]])[0]
            terms["P"].append(value)
            logged["P"].append(float(value.detach()))
            continue
        ref = state_ref(sampling, t)
        origin = encode_ref(ref, phi, perception, sampling, states, action_count)
        record = load_step(REPO / sampling.out_dir, t)
        R = pi.readout(origin.Y, origin.C)[:, -1]
        eligible = (
            kind == "pending"
            or t + ppo.horizon <= (t // sampling.steps + 1) * sampling.steps
        )
        successor = None
        if kind == "fresh":
            successor = encode_ref(
                state_ref(sampling, t + 1),
                phi,
                perception,
                sampling,
                states,
                action_count,
            )
        if eligible:
            logits = pi.logits(R)
            futures = [] if successor is None else [successor.Y]
            futures += [
                encode_ref(
                    state_ref(sampling, t + k),
                    phi,
                    perception,
                    sampling,
                    states,
                    action_count,
                ).Y
                for k in range(len(futures) + 1, ppo.horizon + 1)
            ]
            advantage = reward(P, [origin.Y], [futures], ppo.horizon, discount, min_p)
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
            logged["ppo"].append(float(value.detach()))
        if kind == "pending":
            continue
        actions = torch.tensor([record.action], dtype=torch.int64, device=R.device)
        predicted = pi.shift(R, actions)
        actual = successor.Y[0, -1:] - origin.Y[0, -1:]
        shift = shift_loss(predicted, actual, metric)[0]
        metra = modules["metra"](actual, predicted, alpha)[0]
        terms["shift"].append(shift)
        terms["metra"].append(metra)
        logged["shift"].append(float(shift.detach()))
        logged["metra"].append(float(metra.detach()))
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
) -> dict:
    """Accumulate Eq. (3) into existing gradients, one backward per micro-batch."""
    logged = {role: [] for role in ROLES}
    total_value = 0.0
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
        total_value += float(total.detach())
        for role in ROLES:
            logged[role].extend(values[role])
    result = {
        role: sum(values) / len(values) if values else None
        for role, values in logged.items()
    }
    return result | {"loss": total_value}
