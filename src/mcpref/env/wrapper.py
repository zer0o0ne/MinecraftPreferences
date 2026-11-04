"""Interface Env of specs/env.md: one endless 20 Hz MineStudio timeline."""

import sys

from tqdm import tqdm

from mcpref import repo_path
from mcpref.clips.config import load_config as load_clips_config
from mcpref.engine.config import load_config as load_engine_config
from mcpref.engine.config import use_engine
from mcpref.env.actions import LONG, click, completed, escape, hold, immediate
from mcpref.env.config import EnvConfig
from mcpref.env.state import EnvState, check_episode, convert, observation
from mcpref.gui.mask import load_config as load_mask_config
from mcpref.gui.mask import load_masks
from mcpref.gui.screens import load_screens
from mcpref.recognition.classifier import load_classifier
from mcpref.recognition.classifier import load_config as load_classifier_config
from mcpref.vpt.actions import load_config as load_actions_config

CENTRE = (320.0, 180.0)
CLOSED_AUTOMATICALLY = frozenset({"other", "pause"})


class Env:
    """Interface: construction, one reset, macro-action steps and close."""

    def __init__(self, cfg: EnvConfig):
        engine = load_engine_config(repo_path("engine", cfg.engine))
        clips = load_clips_config(repo_path("clips", cfg.clips))
        actions = load_actions_config(clips.actions)
        mask_cfg = load_mask_config(repo_path("action_mask", cfg.action_mask))
        classifier_cfg = load_classifier_config(repo_path("classifier", cfg.classifier))
        use_engine(engine)
        from mcpref.env.minestudio import construct

        self.masks = load_masks(mask_cfg)
        self.screens = load_screens(repo_path("screens", mask_cfg.screens))
        width, height = engine.render_size
        self.gamma = load_classifier(classifier_cfg, mask_cfg.screens, width, height)[0]
        self.delta, self.turn_deg = clips.delta, actions.turn_deg
        self.cursor_scale = cfg.cursor_scale
        self._engine = engine
        self.sim = construct(engine)
        self.hotbar, self.cursor = 0, CENTRE
        self._raw, self._previous_gui, self._state = None, False, None

    def _accept(self, result: tuple) -> tuple:
        """(4), (8): check and track a raw result, re-centring on GUI transitions."""
        check_episode(result)
        before = False if self._raw is None else bool(self._raw[4]["isGuiOpen"])
        after = bool(result[4]["isGuiOpen"])
        self._previous_gui = before
        if before != after:
            self.cursor = CENTRE
        self._raw = result
        return result

    def _tick(self, q: dict) -> tuple:
        """One raw tick; every caller has just obtained its fresh q from noop_action."""
        return self._accept(self.sim.step(q))

    def _command(self, command: str) -> tuple:
        """One initial rule command, or /kill of the real smoke sequence."""
        from mcpref.env.minestudio import execute

        return self._accept(execute(self.sim, command))

    def reset(self) -> EnvState:
        """(2)-(5): options, the sole world reset, five rule commands, initial state."""
        from mcpref.env.minestudio import COMMANDS, write_options

        write_options(self._engine)
        obs, info = self.sim.reset()
        self._accept((obs, 0.0, False, False, info))
        with tqdm(total=5, desc="env init", unit="command", file=sys.stderr) as bar:
            for command in COMMANDS:
                self._command(command)
                bar.update()
        return self._valid_state(None)

    def _valid_state(self, cause: str | None) -> EnvState:
        """(4), (5): close other/pause before constructing and returning the mask."""
        parts = observation(
            self._raw[0],
            self._raw[4],
            self.hotbar,
            self._previous_gui,
            cause,
            self.gamma,
        )
        while parts[2] and parts[3] in CLOSED_AUTOMATICALLY:
            for pressed in (True, False):
                q = self.sim.noop_action()
                escape(q, pressed)
                self._tick(q)
            parts = observation(
                self._raw[0],
                self._raw[4],
                self.hotbar,
                self._previous_gui,
                None,
                self.gamma,
            )
        self._state = convert(parts, self.masks, self.hotbar)
        return self._state

    def _apply(self, q: dict, action: str, tick: int, finished: bool) -> None:
        """(6)-(9): apply one ordinary tick of the requested macro action."""
        if action in LONG:
            if not finished:
                hold(q, action)
        elif action == "open_inventory":
            if not finished:
                immediate(q, action, self.turn_deg)
        elif action.startswith(("hotbar_", "turn_")):
            if tick == 0:
                immediate(q, action, self.turn_deg)
        elif action == "close":
            if not finished:
                escape(q, tick % 2 == 0)
        else:
            cls = self.screens[self._state.cls]
            self.cursor = click(q, action, tick, cls, self.cursor, self.cursor_scale)

    def step(self, action: int) -> EnvState:
        """(5)-(10): Delta ordinary ticks, then any automatic-close ticks."""
        name = self.masks.names[action]
        if not self._state.mask[action]:
            for _ in range(self.delta):
                self._tick(self.sim.noop_action())
            return self._valid_state(name)
        finished, tick = False, 0
        while tick < self.delta or (name == "close" and not finished):
            before = self._raw[4]
            q = self.sim.noop_action()
            self._apply(q, name, tick, finished)
            result = self._tick(q)
            if name in LONG:
                finished = finished or completed(name, before, result[4])
            elif name == "open_inventory":
                finished = finished or bool(result[4]["isGuiOpen"])
            elif name == "close":
                finished = finished or (tick % 2 == 1 and not result[4]["isGuiOpen"])
            if name.startswith("hotbar_") and tick == 0:
                self.hotbar = int(name.removeprefix("hotbar_")) - 1
            tick += 1
        return self._valid_state(name)

    def close(self) -> None:
        """(1): delegate to MinecraftSim.close."""
        self.sim.close()
