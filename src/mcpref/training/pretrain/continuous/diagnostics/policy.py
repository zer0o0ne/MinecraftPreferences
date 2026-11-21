"""Publish supervised-policy diagnostics from the actual chronological CE events."""

from dataclasses import replace

import torch

from mcpref.diagnostics.core.observations import PolicyObservations
from mcpref.diagnostics.producers.policy import on_policy_observation


def add_policy(record, rows, context, cfg, diagnostics):
    if not rows:
        return
    values = {
        key: torch.cat([row[key].detach().cpu() for row in rows]).float()
        for key in ("logits", "targets", "previous")
    }
    targets = values["targets"]
    prior = targets[targets.sum(dim=-1) > 0].sum(dim=0)
    if prior.sum() > 0:
        prior = prior / prior.sum()
    observation = PolicyObservations(
        **values,
        masks=torch.stack([torch.as_tensor(row["masks"]) for row in rows]),
        actions=torch.tensor([row["action"] for row in rows]),
        prior=prior,
        width=1,
        copy_eps=cfg.action_copy_eps,
        calibration_bins=cfg.calibration_bins,
    )
    producer = replace(
        context,
        producer="pretrain-continuous-policy",
        cluster_ids=tuple(row["source"] for row in rows),
    )
    policy = on_policy_observation(observation, cfg=diagnostics, context=producer)
    record["metrics"].update(policy["metrics"])
    record["populations"].update(policy["populations"])
    record["joined"]["policy"] = policy["joined"]
