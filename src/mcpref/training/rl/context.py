"""Select the perception mode and matching replay contract for an RL run."""

from dataclasses import dataclass

from mcpref.config import REPO
from mcpref.datasets.clips.collect import load_config as load_data
from mcpref.models.perception.config import load_config as load_perception
from mcpref.models.policy.config import load_config as load_policy
from mcpref.models.preference.config import load_config as load_preference
from mcpref.training.pretrain.modes import load_selected_config, selected_config
from mcpref.training.rl.config import ContinuousConfig, RLCycleConfig


@dataclass(frozen=True)
class RLContext:
    mode: str
    pretrain: object
    perception: object
    states: int
    replay: ContinuousConfig | None


def load_context(cfg: RLCycleConfig, sampling) -> RLContext:
    mode, _ = selected_config(REPO / sampling.pretrain)
    pretrain = load_selected_config(REPO / sampling.pretrain)
    perception = load_perception(REPO / pretrain.perception)
    if mode == "continuous":
        if cfg.continuous is None:
            raise ValueError("continuous RL requires the continuous replay config")
        states = max(
            load_policy(REPO / pretrain.policy).history,
            load_preference(REPO / pretrain.preference).history,
        )
        replay = cfg.continuous
    else:
        states = min(
            perception.history, load_data(REPO / sampling.perception_data).states
        )
        replay = None
    return RLContext(mode, pretrain, perception, states, replay)
