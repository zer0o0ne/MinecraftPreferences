"""(1), (2) of specs/env.md: EndlessHumanSurvival and simulator construction."""

from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import (
    HumanSurvival,
)
from minestudio.simulator.minerl.herobraine.hero.handlers.agent.actions.keyboard import (
    KeybasedCommandAction,
)

from mcpref.infrastructure.shards.store import write_file
from mcpref.runtime.engine.config import EngineConfig

COMMANDS = (
    "/gamerule doImmediateRespawn true",
    "/gamerule sendCommandFeedback false",
    "/difficulty medium",
    "/gamerule keepInventory false",
    "/gamerule doDaylightCycle true",
)


class EndlessHumanSurvival(HumanSurvival):
    """(1): HumanSurvival with Escape and without an episode-ending quit producer."""

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
