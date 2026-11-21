"""Bounded recurrent, policy-target and preference suffix state."""

from dataclasses import dataclass, field

import torch


def detached(value, device=None):
    """Copy tensor state without retaining graphs, including nested stream dataclasses."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device or value.device, copy=True)
    if isinstance(value, dict):
        return {key: detached(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached(item, device) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return type(value)(
            **{
                key: detached(getattr(value, key), device)
                for key in value.__dataclass_fields__
            }
        )
    return value


@dataclass
class Stream:
    recurrent: object = None
    previous: dict | None = None
    recent: dict[str, torch.Tensor] = field(default_factory=dict)
    policy: list[dict] = field(default_factory=list)
    pending: dict | None = None

    def detach(self, device=None) -> "Stream":
        result = detached(self, device)
        if result.pending is not None:
            result.pending["readout"] = None
        return result

    def remember(self, key: str, z: torch.Tensor, limit: int) -> None:
        self.recent[key] = z
        while len(self.recent) > limit:
            del self.recent[next(iter(self.recent))]
