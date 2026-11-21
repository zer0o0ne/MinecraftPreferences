"""The pretraining config of the loop tests (specs/pretrain.md Notation: Config) and the module
configs it names (specs/policy.md: `perception`, `action_mask`, `policy`, `preference`),
written under the temporary repository root of loop_run.py."""

from pathlib import Path

import yaml
from models.perception.world import BASE, SPRITE_SHA

MODULE_CONFIGS = {"perception": "w/configs/models/perception.yaml",
                  "action_mask": "w/configs/gui/action-mask.yaml",
                  "policy": "w/configs/models/policy.yaml",
                  "preference": "w/configs/models/preference.yaml"}  # fmt: skip
DIAGNOSTICS = "w/configs/diagnostics.yaml"
SCREENS = "w/configs/screens.yaml"  # the catalogue named by the action-mask config


def write_module_configs(repo: Path) -> None:
    """Test-owned module files, including perception history and the mask catalogue path."""
    for key, written in MODULE_CONFIGS.items():
        (repo / written).parent.mkdir(parents=True, exist_ok=True)
        (repo / written).write_text(f"# {key}\n")
    text = f'screens: "{SCREENS}"\npitch_limit: 90\n'
    (repo / MODULE_CONFIGS["perception"]).write_text(
        yaml.safe_dump(BASE | {"cursor": "w/cursor.png", "cursor_sha256": SPRITE_SHA})
    )
    (repo / MODULE_CONFIGS["action_mask"]).write_text(text)
    (repo / SCREENS).write_text("# screens\n")
    (repo / DIAGNOSTICS).write_text(
        yaml.safe_dump(
            {
                "out_dir": "w/diagnostics",
                "seed": 0,
                "bootstrap_samples": 2,
                "ci_level": 0.95,
                "min_cluster_count": 2,
                "quantiles": [0.05, 0.5, 0.95],
            },
            sort_keys=False,
        )
    )


def action_mask_catalogue() -> dict:
    """A test-owned catalogue whose 90 slots give the 395 action names of the latency/loop tests."""
    elements = [
        {"index": i, "name": f"slot_{i}", "kind": "slot",
         "hit": [0, 0, 18, 18], "click": [9.0, 9.0]}
        for i in range(1, 91)
    ]  # fmt: skip
    elements[0] = elements[0] | {"hit": [239, 238, 18, 18], "click": [248.0, 247.0]}
    member = {"blits": [], "dynamic": [], "labels": []}
    screen = {"members": {"only": member}, "description": "Test screen.",
              "window": [0, 0, 640, 360], "elements": elements}  # fmt: skip
    respawn = {"index": 90, "name": "respawn", "kind": "button",
               "hit": [0, 0, 18, 18], "click": [9.0, 9.0]}  # fmt: skip
    death = screen | {"description": "Test death screen.", "elements": [respawn]}
    textures = {"url": "http://127.0.0.1:9/client.jar", "sha1": "0" * 40,
                "out_dir": "data/textures", "http_timeout": 2.0, "http_retries": 2,
                "retry_delay": 0.05}  # fmt: skip
    return {"classes": {"inventory": screen, "death": death}, "textures": textures}


def write_action_mask_catalogue(repo: Path) -> None:
    doc = action_mask_catalogue()
    (repo / SCREENS).write_text(yaml.safe_dump(doc, sort_keys=False))


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
    "diagnostics": DIAGNOSTICS,
    "seed": 5,
    "epochs": 2,
    "retrain": False,
    "batch": 2,
    "micro_batch": 1,
    "gradient_diagnostics": True,
    "diagnostic_window": 1,
    "reference_steps": 60,
    "reference_pairs": 1,
    "calibration_bins": 10,
    "saturation_threshold": 0.01,
    "inc_classification_threshold": 0.5,
    "quantiles": [0.05, 0.5, 0.95],
    "action_copy_eps": 0.1,
    "latent_init_scale": 1.0,
    "checkpoint_steps": 3,
    "keep_checkpoints": 2,
    "warmup_steps": 3,
    "w_phi": 0.4,
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
