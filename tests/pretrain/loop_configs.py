"""The pretraining config of the loop tests (specs/pretrain.md Notation: Config) and the module
configs it names (specs/policy.md: `perception`, `action_mask`, `policy`, `preference`),
written under the temporary repository root of loop_run.py."""

from pathlib import Path

import yaml

MODULE_CONFIGS = {"perception": "w/configs/perception.yaml",
                  "action_mask": "w/configs/action-mask.yaml",
                  "policy": "w/configs/policy.yaml",
                  "preference": "w/configs/preference.yaml"}  # fmt: skip
SCREENS = "w/configs/screens.yaml"  # the catalogue named by the action-mask config


def write_module_configs(repo: Path) -> None:
    """The files named by MODULE_CONFIGS and SCREENS, each of its own bytes: `pretrain` only
    hashes them and reads `screens` of the action-mask config (specs/pretrain.md (R) 6)."""
    for key, written in MODULE_CONFIGS.items():
        (repo / written).parent.mkdir(parents=True, exist_ok=True)
        (repo / written).write_text(f"# {key}\n")
    text = f'screens: "{SCREENS}"\npitch_limit: 90\n'
    (repo / MODULE_CONFIGS["action_mask"]).write_text(text)
    (repo / SCREENS).write_text("# screens\n")


GROUPS = {  # in another order than the groups' (phi_A, phi_c, dual, pi, idle, pref)
    "pref": {"lr": 0.04, "weight_decay": 0.02},
    "pi": {"lr": 0.03, "weight_decay": 0.01},
    "idle": {"lr": 0.2, "weight_decay": 0.5},
    "dual": {"lr": 0.1, "weight_decay": 0.0},
    "phi_c": {"lr": 0.02, "weight_decay": 0.0},
    "phi_A": {"lr": 0.05, "weight_decay": 0.1},
}
CONFIG = {
    "index_dir": "w/index",
    "out_dir": "w/pretrain",
    **MODULE_CONFIGS,
    "seed": 5,
    "epochs": 2,
    "batch": 2,
    "micro_batch": 1,
    "checkpoint_steps": 3,
    "keep_checkpoints": 2,
    "warmup_steps": 3,
    "w_pi": 0.7,
    "w_P": 1.9,
    "grad_clip": 1.0,
    "adam_eps": 1.0e-4,
    "lr_floor": 0.2,
    "betas": [0.8, 0.95],
    "groups": GROUPS,
    "device": "cpu",
}
NO_PREF = CONFIG | {"groups": {k: v for k, v in GROUPS.items() if k != "pref"}}


def config_text(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False)
