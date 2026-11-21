"""Training observations and the shared diagnostics integration hooks."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Context:
    cfg: object
    modules: dict
    index: object
    root: Path
    groups: dict
    optimizer: object
    total: int
    phase: str
    step: int = 0
    epoch: int = 1
    fragment: dict | None = None


def notify(hooks, name: str, *args) -> None:
    callback = getattr(hooks, name, None)
    if callback is not None:
        callback(*args)


@dataclass
class WindowResult:
    weighted: dict = field(default_factory=dict)
    sums: dict = field(default_factory=dict)
    counts: dict = field(
        default_factory=lambda: dict.fromkeys(("phi", "ce", "shift", "P"), 0)
    )
    phi_rows: list = field(default_factory=list)
    ce_rows: list = field(default_factory=list)
    shift_rows: list = field(default_factory=list)
    preference_rows: list = field(default_factory=list)
    gradients: dict | None = None
    updates: dict | None = None
    lr: dict = field(default_factory=dict)

    def add(self, group: str, terms: dict) -> None:
        for key, value in terms.items():
            name = f"{group}_{key}"
            self.sums[name] = self.sums.get(name, 0) + value.detach()

    def record(self, step: int) -> dict:
        scalar = lambda x: float(x.detach().cpu()) if isinstance(x, torch.Tensor) else x
        return {
            "step": step,
            "counts": dict(self.counts),
            "components": {key: scalar(value) for key, value in self.sums.items()},
            "weighted": {key: scalar(value) for key, value in self.weighted.items()},
            "loss": sum(scalar(value) for value in self.weighted.values()),
            "lr": self.lr,
            "gradients": self.gradients,
            "updates": self.updates,
        }


def backward(context, hooks, components: dict, events: dict, *, retain: bool) -> None:
    differentiable = {
        key: value for key, value in components.items() if value.requires_grad
    }
    if not differentiable:
        return
    notify(hooks, "before_backward", context, differentiable, events)
    sum(differentiable.values()).backward(retain_graph=retain)
