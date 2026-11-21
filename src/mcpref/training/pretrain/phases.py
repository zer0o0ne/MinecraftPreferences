"""Shared frozen-perception phase setup."""

from pathlib import Path

import torch

from mcpref.training.groups import Groups, check_groups
from mcpref.training.pretrain.config import TrainingConfig


def head_groups(cfg: TrainingConfig, modules: dict) -> Groups:
    declared = check_groups(cfg, modules)
    heads = {
        id(parameter)
        for name in ("pi", "P")
        if name in modules
        for parameter in modules[name].parameters()
    }
    selected = Groups()
    selected.parameter_names = declared.parameter_names
    for name, parameters in declared.items():
        group = [parameter for parameter in parameters if id(parameter) in heads]
        if group:
            selected[name] = group
    return selected


def load_joint_model(path: Path, modules: dict) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(state[name])
