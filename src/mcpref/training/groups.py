"""Declaration and validation of optimizer parameter groups."""

import torch

from mcpref.training.checkpoints import Modules
from mcpref.training.pretrain.config import PretrainConfig


class Groups(dict[str, list[torch.nn.Parameter]]):
    """Groups in declaration order with canonical module-qualified parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.parameter_names: dict[int, str] = {}


def declared(modules: Modules) -> Groups:
    """(R) 4: the groups by module, then by declaration; a name twice aborts."""
    groups = Groups()
    for name, module in modules.items():
        for parameter_name, parameter in module.named_parameters():
            groups.parameter_names[id(parameter)] = f"{name}.{parameter_name}"
        for q, params in module.groups().items():
            if q in groups:
                raise RuntimeError(f"group {q} declared twice")
            groups[q] = params
    return groups


def check_groups(cfg: PretrainConfig, modules: Modules) -> Groups:
    """(R) 4: names once, each parameter in one group, every parameter in a group, the
    names those of the config."""
    groups, owner = declared(modules), {}
    for q, params in groups.items():
        for p in params:
            if id(p) in owner:
                raise RuntimeError(
                    f"a parameter is twice in the group {q}"
                    if owner[id(p)] == q
                    else f"a parameter is in the groups {owner[id(p)]} and {q}"
                )
            owner[id(p)] = q
    for name, module in modules.items():
        if any(id(p) not in owner for p in module.parameters()):
            raise RuntimeError(f"{name} has parameters outside the groups")
    names, configured = list(groups), list(cfg.groups)
    if set(names) != set(configured):
        raise RuntimeError(
            f"the modules declare the groups {names}, the config {configured}"
        )
    return groups
