"""Fake MinecraftSim and hand-built raw results of specs/env.md (Notation, (3)-(9))."""

import copy
from collections import deque

import numpy as np

FRAME = np.zeros((360, 640, 3), dtype=np.uint8)
JUMP = "minecraft.custom:minecraft.jump"
DAMAGE = "minecraft.custom:minecraft.damage_dealt"
MINE = "minecraft.mine_block:minecraft.stone"
USE = "minecraft.use_item:minecraft.stone"


def noop() -> dict:
    action = {
        "attack": 0,
        "back": 0,
        "forward": 0,
        "jump": 0,
        "left": 0,
        "right": 0,
        "sneak": 0,
        "sprint": 0,
        "use": 0,
        "inventory": 0,
        "camera": np.zeros(2, dtype=np.float32),
        "ESC": 0,
        "chat": "",
        "drop": 0,
        "swapHands": 0,
        "pickItem": 0,
        "F1": 0,
        "F3": 0,
        "F5": 0,
    }
    action.update({f"hotbar.{j}": 0 for j in range(1, 10)})
    return action


def expected(**changes) -> dict:
    action = noop()
    action.update(changes)
    return action


def raw(
    *,
    gui: bool = False,
    frame: np.ndarray = FRAME,
    pos: dict | None = None,
    inventory: dict | None = None,
    custom: dict | None = None,
    mine_block: dict | None = None,
    use_item: dict | None = None,
    terminated: bool = False,
    truncated: bool = False,
    error: str | None = None,
) -> tuple:
    info = {
        "player_pos": pos or {"x": 1, "y": 2, "z": 3, "yaw": 4, "pitch": 5},
        "inventory": inventory or {},
        "isGuiOpen": gui,
        "custom": custom or {},
        "mine_block": mine_block or {},
        "use_item": use_item or {},
    }
    if error is not None:
        info["error"] = error
    return {"image": frame}, 17.0, terminated, truncated, info


class FakeSimulator:
    """One current raw result, replaced by queued results; every action is freshly recorded."""

    def __init__(self):
        self.current = raw()
        self.reset_raw = raw()
        self.results = deque()
        self.actions = []
        self.original_actions = []
        self.action_ids = []
        self.camera_ids = []
        self.commands = []
        self.reset_calls = 0
        self.close_calls = 0
        self.init_kwargs = None

    def reset(self):
        self.reset_calls += 1
        self.current = self.reset_raw
        obs, _, _, _, info = self.current
        return obs, info

    def noop_action(self):
        return noop()

    def step(self, action):
        self.original_actions.append(action)
        self.actions.append(copy.deepcopy(action))
        self.action_ids.append(id(action))
        self.camera_ids.append(id(action["camera"]))
        if self.results:
            self.current = self.results.popleft()
        return self.current

    def close(self):
        self.close_calls += 1

    @property
    def env(self):
        return self

    def execute_cmd(self, command):
        self.commands.append(command)
        obs, reward, _, _, info = self.current
        return obs, reward, False, info

    def _wrap_obs_info(self, obs, info):
        return obs, info


def assert_actions(fake: FakeSimulator, wanted: list[dict]) -> None:
    assert len(fake.actions) == len(wanted)
    for got, expected_action in zip(fake.actions, wanted, strict=True):
        assert got.keys() == expected_action.keys()
        for key in got:
            if key == "camera":
                np.testing.assert_array_equal(got[key], expected_action[key])
                assert got[key].dtype == np.float32
            else:
                assert got[key] == expected_action[key], key
    assert len(set(fake.action_ids)) == len(fake.actions)
    assert len(set(fake.camera_ids)) == len(fake.actions)
