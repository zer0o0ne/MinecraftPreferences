"""(1), (2) of specs/env.md: EndlessHumanSurvival and simulator construction."""

import numpy as np
from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import (
    HumanSurvival,
)
from minestudio.simulator.minerl.herobraine.hero import spaces
from minestudio.simulator.minerl.herobraine.hero.handlers.agent.actions.keyboard import (
    KeybasedCommandAction,
)
from minestudio.simulator.minerl.herobraine.hero.handlers.translation import (
    TranslationHandler,
)

from mcpref.infrastructure.shards.store import write_file
from mcpref.runtime.engine.config import EngineConfig
from mcpref.runtime.env.progress import action_progress

COMMANDS = (
    "/gamerule doImmediateRespawn true",
    "/gamerule sendCommandFeedback false",
    "/difficulty medium",
    "/gamerule keepInventory false",
    "/gamerule doDaylightCycle true",
)


class EndlessHumanSurvival(HumanSurvival):
    """(1): HumanSurvival with Escape and without an episode-ending quit producer."""

    def create_observables(self) -> list:
        """Include actual mining and item-use progress on every raw tick."""
        return super().create_observables() + [ActionProgress()]

    def create_actionables(self) -> list:
        """(1): Escape registered after every ordinary actionable."""
        return super().create_actionables() + [KeybasedCommandAction("ESC", "ESC")]

    def create_server_quit_producers(self) -> list:
        """(1), (10): no server quit producers on the endless timeline."""
        return []


def construct(cfg: EngineConfig):
    """(1): MinecraftSim with HumanSurvival replaced only during its construction."""
    from minestudio.simulator import entry

    original = entry.HumanSurvival
    entry.HumanSurvival = EndlessHumanSurvival
    try:
        return entry.MinecraftSim(
            action_type="env",
            obs_size=cfg.render_size,
            render_size=cfg.render_size,
            seed=cfg.seed,
        )
    finally:
        entry.HumanSurvival = original


def write_options(cfg: EngineConfig) -> None:
    """(2): Auto-Jump on, all other options left to the game defaults."""
    write_file(cfg.minestudio_dir / "runtime" / "options.txt", b"autoJump:true\n")


def execute(sim, command: str) -> tuple:
    """(2): one execute_cmd, adapted to MinecraftSim's wrapped step-result format."""
    obs, _, done, info = sim.env.execute_cmd(command)
    obs, info = sim._wrap_obs_info(obs, info)
    return obs, 0.0, done, done, info


class ActionProgress(TranslationHandler):
    """Expose the additional JSON object emitted by the patched engine."""

    def __init__(self):
        super().__init__(
            spaces.Dict(
                {
                    "version": spaces.Discrete(2),
                    "mining": spaces.Dict(
                        {
                            "active": spaces.Discrete(2),
                            "target": spaces.Box(
                                low=-(2**31), high=2**31 - 1, shape=(3,), dtype=np.int64
                            ),
                            "progress": spaces.Box(
                                low=0, high=1, shape=(), dtype=np.float64
                            ),
                        }
                    ),
                    "item_use": spaces.Dict(
                        {
                            "active": spaces.Discrete(2),
                            "action": spaces.Enum(
                                "NONE",
                                "EAT",
                                "DRINK",
                                "BLOCK",
                                "BOW",
                                "SPEAR",
                                "CROSSBOW",
                            ),
                            "duration": spaces.Box(
                                low=0, high=2**31 - 1, shape=(), dtype=np.int64
                            ),
                            "charge_duration": spaces.Box(
                                low=0, high=2**31 - 1, shape=(), dtype=np.int64
                            ),
                            "elapsed": spaces.Box(
                                low=0, high=2**31 - 1, shape=(), dtype=np.int64
                            ),
                            "remaining": spaces.Box(
                                low=-(2**31), high=2**31 - 1, shape=(), dtype=np.int64
                            ),
                        }
                    ),
                }
            )
        )

    def to_string(self) -> str:
        return "action_progress"

    def xml_template(self) -> str:
        return ""

    def from_hero(self, info: dict) -> dict:
        return action_progress(info)
