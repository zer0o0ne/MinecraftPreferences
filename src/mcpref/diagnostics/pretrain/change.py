"""Pooled head agreement in specs/action-change-diagnostics.md."""

import torch

METRICS = (
    "repeat_when_predicted",
    "head_recall",
    "action_recall",
    "joint_recall",
    "false_positive",
)


def counts(logits, q, previous, probability, known):
    """Return detached numerator/denominator pairs, without retaining rows."""
    valid = known.detach() & (q.detach().sum(-1) > 0)
    targets = q.detach()[valid].double()
    targets = targets / targets.sum(-1, keepdim=True)
    top = logits.detach()[valid].argmax(-1, keepdim=True)
    old = previous.detach()[valid] > 0
    repeated = old.gather(-1, top).squeeze(-1)
    correct_change = ~repeated & (targets.gather(-1, top).squeeze(-1) > 0)
    predicted = (probability.detach()[valid] >= 0.5).double()
    change = (targets * ~old).sum(-1)
    repeat = (targets * old).sum(-1)
    return torch.stack(
        [
            torch.stack((numerator.sum(), denominator.sum()))
            for numerator, denominator in (
                (predicted * repeated, predicted),
                (change * predicted, change),
                (change * correct_change, change),
                (change * predicted * correct_change, change),
                (repeat * predicted, repeat),
            )
        ]
    )


def pool(values):
    values = [value for value in values if value is not None]
    return torch.stack(values).sum(0) if values else None


def suite(value):
    if value is None:
        return {}
    rows = value.detach().cpu().tolist()
    return {
        "count": round(rows[1][1] + rows[4][1]),
        **{
            name: {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator else None,
            }
            for name, (numerator, denominator) in zip(METRICS, rows)
        },
    }


def join_metrics(record, outputs):
    """Publish pooled checkpoint statistics, without claiming row-level clusters."""
    for name, value in outputs.get("pi_action", {}).get("change", {}).items():
        key = f"policy.change.{name}"
        record["metrics"][key] = value
        record["populations"][key] = []
