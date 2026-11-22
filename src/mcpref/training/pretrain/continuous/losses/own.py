"""The unchanged perception bracket and separated policy CE/shift events."""

import torch
from torch.nn import functional as F

from mcpref.models.perception.loss import _terms, changed
from mcpref.models.policy.targets import frequencies


def phi_interval(module, previous: dict, z, observation):
    cfg = module.cfg
    mask = changed(
        [previous["frame"], observation.frame],
        [previous["meta"], observation.meta],
        cfg.tau,
    )
    return _terms(
        torch.stack((previous["z"], z)),
        z.new_tensor([previous["interval"]["n"]]),
        torch.tensor(mask, device=z.device),
        module.log_lambda,
        module.metric,
        cfg.c,
        cfg.beta,
        cfg.dual_slack,
    )


def action_input(pi, interval, z):
    actions = interval["actions"] if interval and interval["known"] else []
    c = z.new_tensor(frequencies(actions, pi.index, len(pi.names) + 1))
    if not actions:
        c[-1] = 1
    return c


def eligible(observation) -> bool:
    return bool(observation.interval and observation.interval["valid"])


def policy_ce(pi, stream, observation, z):
    interval = observation.interval
    kept = [a for a in interval["actions"] if observation.mask[pi.index[a]]]
    q = z.new_tensor(frequencies(kept, pi.index, len(pi.names)))
    Y = torch.stack([row["z"] for row in stream.policy])[None]
    C = torch.stack([row["c"] for row in stream.policy])[None]
    readout = pi.readout(Y, C)[:, -1]
    logits = pi.logits(readout)
    ce = F.cross_entropy(logits, q[None], reduction="sum", label_smoothing=0.0)
    valid = interval["n"] == len(kept) == 1
    action = pi.index[interval["actions"][0]] if interval["n"] == 1 else 0
    pending = {
        "id": observation.id,
        "target": interval["b"],
        "z": z,
        "Y": Y,
        "C": C,
        "readout": readout,
        "action": action,
        "valid": valid,
        "indices": [row["id"] for row in stream.policy],
    }
    row = {
        "id": observation.id,
        "logits": logits.detach(),
        "targets": q[None].detach(),
        "previous": C[:, -1, :-1].detach(),
        "masks": observation.mask,
        "action": action,
        "action_valid": bool(kept),
        "shift_valid": valid,
        "source": observation.source,
    }
    return ce, pending, row


def policy_shift(pi, loss, pending, z):
    readout = pending["readout"]
    if readout is None:
        readout = pi.readout(pending["Y"], pending["C"])[:, -1]
    a = torch.tensor([pending["action"]], dtype=torch.int64, device=z.device)
    prediction = pi.shift(readout, a)
    factual = (z - pending["z"]).detach()[None]
    d2 = loss.metric(prediction - factual).square().sum()
    zero = loss.metric(factual).square().sum()
    valid = torch.tensor(pending["valid"], device=z.device)
    return (
        loss.shift_weight * torch.where(valid, d2, 0.0),
        loss.shift_weight * torch.where(valid, zero, 0.0),
    )
